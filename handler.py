import os
import io
import base64
import traceback

import torch
import runpod

from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# Configuration
# ============================================================

MODEL_ID = os.getenv(
    "MODEL_ID",
    "Qingyun/RSCoVLM-7B-2512"
)

MAX_NEW_TOKENS = int(
    os.getenv("MAX_NEW_TOKENS", "512")
)

MIN_PIXELS = int(
    os.getenv("MIN_PIXELS", str(256 * 28 * 28))
)

MAX_PIXELS = int(
    os.getenv("MAX_PIXELS", str(1280 * 28 * 28))
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Global model objects
# Loaded once when the worker starts.
# ============================================================

model = None
processor = None


# ============================================================
# Model loading
# ============================================================

def load_model():

    global model
    global processor

    print("=" * 70)
    print("SNZ RSCoVLM WORKER STARTING")
    print("=" * 70)

    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for the RSCoVLM worker."
        )

    print(
        f"GPU count: {torch.cuda.device_count()}"
    )

    for index in range(torch.cuda.device_count()):
        print(
            f"GPU {index}: "
            f"{torch.cuda.get_device_name(index)}"
        )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    print("-" * 70)
    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print("Processor loaded successfully.")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("-" * 70)
    print("Loading RSCoVLM model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("RSCoVLM loaded successfully.")

    if torch.cuda.is_available():

        print(
            "GPU memory allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        print(
            "GPU memory reserved: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )

    print("=" * 70)


# ============================================================
# Base64 image decoding
# ============================================================

def decode_base64_image(image_b64: str) -> Image.Image:

    if not image_b64:
        raise ValueError(
            "Image Base64 data is empty."
        )

    # Support:
    # data:image/png;base64,...
    # data:image/jpeg;base64,...
    # raw Base64
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(
            ",",
            1
        )[1]

    try:

        image_bytes = base64.b64decode(
            image_b64,
            validate=True
        )

    except Exception as exc:

        raise ValueError(
            f"Invalid Base64 image data: {exc}"
        )

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as exc:

        raise ValueError(
            f"Unable to decode image: {exc}"
        )

    return image


# ============================================================
# Collect images from the RunPod request
#
# Supported formats:
#
# 1. Single image:
#    image_b64
#
# 2. Multiple images:
#    images_b64: [image1, image2, ...]
#
# 3. Bi-temporal:
#    image_t1_b64
#    image_t2_b64
#
# The function also prevents duplicate images when the
# caller accidentally supplies more than one format.
# ============================================================

def collect_images(job_input: dict):

    images = []
    image_labels = []

    # --------------------------------------------------------
    # Format 1:
    # image_b64
    # --------------------------------------------------------

    image_b64 = job_input.get(
        "image_b64"
    )

    if image_b64:
        images.append(
            decode_base64_image(image_b64)
        )

        image_labels.append(
            "Primary image"
        )

    # --------------------------------------------------------
    # Format 2:
    # images_b64
    # --------------------------------------------------------

    images_b64 = job_input.get(
        "images_b64"
    )

    if images_b64:

        if not isinstance(images_b64, list):
            raise ValueError(
                "'images_b64' must be a list of Base64 images."
            )

        for index, image_data in enumerate(
            images_b64
        ):

            if not image_data:
                continue

            images.append(
                decode_base64_image(
                    image_data
                )
            )

            image_labels.append(
                f"Image {index + 1}"
            )

    # --------------------------------------------------------
    # Format 3:
    # image_t1_b64 + image_t2_b64
    #
    # This format is useful for temporal pairs.
    # --------------------------------------------------------

    image_t1_b64 = job_input.get(
        "image_t1_b64"
    )

    image_t2_b64 = job_input.get(
        "image_t2_b64"
    )

    if image_t1_b64:

        images.append(
            decode_base64_image(
                image_t1_b64
            )
        )

        image_labels.append(
            "Time 1 image"
        )

    if image_t2_b64:

        images.append(
            decode_base64_image(
                image_t2_b64
            )
        )

        image_labels.append(
            "Time 2 image"
        )

    if not images:

        raise ValueError(
            "No image was supplied. "
            "Provide image_b64, images_b64, "
            "or image_t1_b64/image_t2_b64."
        )

    return images, image_labels


# ============================================================
# Prompt construction
# ============================================================

def build_prompt(
    prompt: str,
    image_count: int,
    image_labels=None,
    conversation_history=None,
    metadata=None,
):

    prompt = (prompt or "").strip()

    if not prompt:

        prompt = (
            "Analyze the supplied remote sensing image."
        )

    # --------------------------------------------------------
    # Single vs multiple image instructions
    # --------------------------------------------------------

    if image_count == 1:

        image_instruction = """
You are analyzing one remote-sensing image.

Base your answer on the visible evidence in that image.
""".strip()

    else:

        image_instruction = f"""
You are analyzing {image_count} remote-sensing images.

Treat the supplied images as separate but related
observations.

Compare them when the user's question requires comparison.

Do not assume that multiple images represent the same
location or time unless the supplied metadata or user
question indicates this.

When comparing images, explicitly distinguish:
- observations common to the images
- differences between the images
- uncertain interpretations
""".strip()

    # --------------------------------------------------------
    # System instruction
    # --------------------------------------------------------

    system_instruction = f"""
You are RSCoVLM, a remote-sensing
vision-language assistant.

{image_instruction}

Analyze the supplied remote-sensing imagery carefully.

Answer the user's question using information supported
by the supplied imagery.

When appropriate:

- identify land-cover or scene characteristics
- describe visible objects
- describe spatial relationships
- identify built-up areas, roads, water, vegetation,
  agricultural regions, infrastructure, or other
  visible features
- compare supplied images when relevant
- distinguish observations from uncertain interpretations
- do not invent geographic facts that cannot be supported
  by the supplied imagery

Give a concise, useful answer suitable for an analytical
remote-sensing application.
""".strip()

    # --------------------------------------------------------
    # Image labels
    # --------------------------------------------------------

    labels_text = ""

    if image_labels:

        labels_text = (
            "\n\nImage ordering:\n"
            + "\n".join(
                f"{index + 1}. {label}"
                for index, label in enumerate(
                    image_labels
                )
            )
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_text = ""

    if metadata:

        metadata_text = (
            "\n\nImage metadata supplied by the application:\n"
            f"{metadata}"
        )

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    history_text = ""

    if conversation_history:

        recent = conversation_history[-6:]

        history_lines = []

        for item in recent:

            role = item.get(
                "role",
                "user"
            )

            content = item.get(
                "content",
                ""
            )

            if content:

                history_lines.append(
                    f"{role}: {content}"
                )

        if history_lines:

            history_text = (
                "\n\nRecent conversation context:\n"
                + "\n".join(history_lines)
            )

    # --------------------------------------------------------
    # Final prompt
    # --------------------------------------------------------

    final_prompt = (
        system_instruction
        + labels_text
        + metadata_text
        + history_text
        + "\n\nUser question:\n"
        + prompt
    )

    return final_prompt


# ============================================================
# Multi-image RSCoVLM inference
# ============================================================

@torch.inference_mode()
def run_inference(
    images,
    prompt: str,
):

    # --------------------------------------------------------
    # Construct a single user message containing ALL images.
    #
    # This is the important change from the previous
    # single-image implementation.
    # --------------------------------------------------------

    content = []

    for image in images:

        content.append(
            {
                "type": "image",
                "image": image,
            }
        )

    # Text is placed after all images.
    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # --------------------------------------------------------
    # Chat template
    # --------------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # --------------------------------------------------------
    # Extract visual inputs
    # --------------------------------------------------------

    image_inputs, video_inputs = (
        process_vision_info(
            messages
        )
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # Move tensors to model device
    # --------------------------------------------------------

    inputs = inputs.to(
        model.device
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
    )

    # --------------------------------------------------------
    # Remove prompt tokens
    # --------------------------------------------------------

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(
            inputs.input_ids,
            generated_ids
        )
    ]

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if not output_text:
        return ""

    return output_text[0].strip()


# ============================================================
# RunPod handler
# ============================================================

def handler(job):

    job_input = job.get(
        "input",
        {}
    )

    try:

        # ----------------------------------------------------
        # Request information
        # ----------------------------------------------------

        prompt = job_input.get(
            "prompt",
            "Analyze the supplied remote sensing imagery."
        )

        metadata = job_input.get(
            "metadata",
            {}
        )

        conversation_history = job_input.get(
            "conversation_history",
            []
        )

        # ----------------------------------------------------
        # Collect one or multiple images
        # ----------------------------------------------------

        images, image_labels = collect_images(
            job_input
        )

        image_count = len(images)

        # ----------------------------------------------------
        # Log request
        # ----------------------------------------------------

        print("=" * 70)

        print(
            f"RSCoVLM JOB RECEIVED | "
            f"{image_count} IMAGE(S)"
        )

        print(
            f"Prompt: {prompt}"
        )

        for index, image in enumerate(images):

            print(
                f"{image_labels[index]}: "
                f"{image.width}x{image.height}"
            )

        print("=" * 70)

        # ----------------------------------------------------
        # Build application-aware prompt
        # ----------------------------------------------------

        final_prompt = build_prompt(
            prompt=prompt,
            image_count=image_count,
            image_labels=image_labels,
            conversation_history=conversation_history,
            metadata=metadata,
        )

        # ----------------------------------------------------
        # Run RSCoVLM
        # ----------------------------------------------------

        response_text = run_inference(
            images=images,
            prompt=final_prompt,
        )

        if not response_text:

            response_text = (
                "RSCoVLM completed inference "
                "but returned an empty response."
            )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "response_text": response_text,

            "groundings": [],

            "model_used": MODEL_ID,

            "metadata": {
                "image_count": image_count,
                "images": [
                    {
                        "label": image_labels[index],
                        "width": image.width,
                        "height": image.height,
                    }
                    for index, image in enumerate(images)
                ],
                "device": DEVICE,
            },
        }

    except Exception as exc:

        print("=" * 70)
        print("RSCoVLM INFERENCE ERROR")
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70)

        return {
            "response_text": "",
            "groundings": [],
            "model_used": MODEL_ID,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# Start worker
# ============================================================

if __name__ == "__main__":

    load_model()

    print(
        "Starting RunPod serverless worker..."
    )

    runpod.serverless.start(
        {
            "handler": handler
        }
    )