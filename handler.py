
import os
import io
import re
import base64
import traceback
import math

import torch
import runpod

from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIGURATION
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

model = None
processor = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    global model, processor

    print("=" * 60)
    print("SNZ RSCoVLM WORKER STARTING")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for RSCoVLM."
        )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("RSCoVLM loaded successfully.")


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_base64_image(image_b64):
    if not image_b64:
        raise ValueError("Image Base64 data is empty.")

    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]

    image_bytes = base64.b64decode(
        image_b64,
        validate=True
    )

    return Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_images(job_input):
    images = []
    labels = []

    image_b64 = job_input.get("image_b64")

    if image_b64:
        images.append(
            decode_base64_image(image_b64)
        )
        labels.append("Primary image")

    images_b64 = job_input.get("images_b64")

    if images_b64:
        if not isinstance(images_b64, list):
            raise ValueError(
                "'images_b64' must be a list."
            )

        for index, image_data in enumerate(images_b64):
            if image_data:
                images.append(
                    decode_base64_image(image_data)
                )
                labels.append(
                    f"Image {index + 1}"
                )

    # Compatibility with the existing agent.py payload.
    additional_images = job_input.get(
        "additional_images_b64"
    )

    if additional_images:
        if not isinstance(additional_images, list):
            raise ValueError(
                "'additional_images_b64' must be a list."
            )

        for index, image_data in enumerate(additional_images):
            if image_data:
                images.append(
                    decode_base64_image(image_data)
                )
                labels.append(
                    f"Additional image {index + 1}"
                )

    # Temporal pair fields are retained for compatibility.
    image_t1 = job_input.get("image_t1_b64")
    image_t2 = job_input.get("image_t2_b64")

    if image_t1:
        images.append(
            decode_base64_image(image_t1)
        )
        labels.append("Time 1 image")

    if image_t2:
        images.append(
            decode_base64_image(image_t2)
        )
        labels.append("Time 2 image")

    if not images:
        raise ValueError(
            "No image supplied to RSCoVLM."
        )

    return images, labels


# ============================================================
# TASK TYPE
# ============================================================

def detect_task_type(job_input):
    task_type = (
        job_input.get("task_type", "")
        .strip()
        .lower()
    )

    if task_type in ("caption", "grounding", "vqa"):
        return task_type

    task_name = (
        job_input.get("task_name", "")
        .strip()
        .lower()
    )

    prompt = (
        job_input.get("prompt", "")
        .strip()
        .lower()
    )

    text = f"{task_name} {prompt}"

    caption_terms = (
        "caption",
        "describe the scene",
        "scene description",
        "describe this image",
        "summarize the image",
    )

    grounding_terms = (
        "grounding",
        "bounding box",
        "bounding boxes",
        "bbox",
        "locate",
        "mark the",
        "highlight the",
        "detect objects",
        "object detection",
    )

    if any(term in text for term in caption_terms):
        return "caption"

    if any(term in text for term in grounding_terms):
        return "grounding"

    return "vqa"


# ============================================================
# TASK-SPECIFIC INSTRUCTIONS
# ============================================================

def task_instruction(task_type, image):
    width, height = image.size

    if task_type == "caption":
        return f"""
TASK: IMAGE CAPTIONING

Image dimensions: width={width}, height={height} pixels.

Write a concise, informative caption describing the
visible remote-sensing scene.

Mention relevant visible features such as buildings,
roads, vegetation, water, agricultural land, or infrastructure.

Do not output bounding boxes.
Do not invent geographic locations or unsupported details.
""".strip()

    if task_type == "grounding":
        return f"""
TASK: VISUAL GROUNDING

Image dimensions: width={width}, height={height} pixels.

Identify the object or feature requested by the user.

When you can identify it, output its bounding box using
this exact format:

x1,y1,x2,y2 - object label

Coordinates must be pixel coordinates in the supplied image:
x1 = left, y1 = top, x2 = right, y2 = bottom.

The valid coordinate range is:
x: 0 to {width}
y: 0 to {height}

Use one object per line.
Do not use geographic longitude or latitude.
Do not invent a box if the object cannot be located.
Briefly state if the requested object is not identifiable.
""".strip()

    return """
TASK: REMOTE-SENSING VISUAL QUESTION ANSWERING

Answer the user's question using visible image evidence.
Be concise and distinguish observations from uncertainty.
Do not invent geographic facts.
""".strip()


# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def build_prompt(
    prompt,
    task_type,
    image,
    image_count,
    image_labels=None,
    conversation_history=None,
    metadata=None,
):
    prompt = (prompt or "").strip()

    if not prompt:
        prompt = "Analyze the supplied remote-sensing image."

    instruction = task_instruction(
        task_type,
        image
    )

    labels_text = ""

    if image_labels:
        labels_text = (
            "\n\nImage ordering:\n"
            + "\n".join(
                f"{i + 1}. {label}"
                for i, label in enumerate(image_labels)
            )
        )

    metadata_text = ""

    if metadata:
        metadata_text = (
            "\n\nApplication image metadata:\n"
            + str(metadata)
        )

    history_text = ""

    if conversation_history:
        recent = conversation_history[-6:]
        lines = []

        for item in recent:
            role = item.get("role", "user")
            content = item.get("content", "")

            if content:
                lines.append(
                    f"{role}: {content}"
                )

        if lines:
            history_text = (
                "\n\nRecent conversation:\n"
                + "\n".join(lines)
            )

    return (
        "You are RSCoVLM, a remote-sensing vision-language assistant.\n\n"
        + instruction
        + labels_text
        + metadata_text
        + history_text
        + "\n\nUser request:\n"
        + prompt
    )


# ============================================================
# PARSE PIXEL BOXES
# ============================================================

def parse_groundings(
    response_text,
    image_width,
    image_height
):
    """
    Parses RSCoVLM outputs such as:

        1245,430,1297,481 visible buildings
        1245,430,1297,481 - visible buildings
        (1245,430,1297,481) - visible buildings

    Assumes raw coordinates are x1,y1,x2,y2 pixels.

    Returns normalized boxes in frontend order:
        [ymin, xmin, ymax, xmax]
    """

    groundings = []

    if not response_text:
        return groundings

    if image_width <= 0 or image_height <= 0:
        return groundings

    pattern = re.compile(
        r"""
        ^\s*
        \(?\s*
        (-?\d+(?:\.\d+)?)\s*[,;]\s*
        (-?\d+(?:\.\d+)?)\s*[,;]\s*
        (-?\d+(?:\.\d+)?)\s*[,;]\s*
        (-?\d+(?:\.\d+)?)\s*
        \)?
        \s*
        (?:[-:|]\s*)?
        (.*?)
        \s*$
        """,
        re.VERBOSE
    )

    for line in response_text.splitlines():
        match = pattern.fullmatch(line)

        if not match:
            continue

        try:
            x1, y1, x2, y2 = [
                float(match.group(i))
                for i in range(1, 5)
            ]
        except (TypeError, ValueError):
            continue

        if not all(
            math.isfinite(value)
            for value in (x1, y1, x2, y2)
        ):
            continue

        label = (
            match.group(5) or "Detected object"
        ).strip()

        if not label:
            label = "Detected object"

        # Reject impossible or out-of-image coordinates.
        if (
            x1 < 0 or y1 < 0
            or x2 < 0 or y2 < 0
            or x1 > image_width
            or x2 > image_width
            or y1 > image_height
            or y2 > image_height
        ):
            continue

        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)

        if right <= left or bottom <= top:
            continue

        # Normalize for ImageBinder:
        # [ymin, xmin, ymax, xmax]
        bbox = [
            top / image_height,
            left / image_width,
            bottom / image_height,
            right / image_width,
        ]

        groundings.append({
            "bbox": bbox,
            "label": label,
            "coordinate_type": "normalized_image_pixels",
        })

    return groundings


# ============================================================
# MODEL INFERENCE
# ============================================================

@torch.inference_mode()
def run_inference(images, prompt):
    content = []

    for image in images:
        content.append({
            "type": "image",
            "image": image,
        })

    content.append({
        "type": "text",
        "text": prompt,
    })

    messages = [{
        "role": "user",
        "content": content,
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
    )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return (
        output_text[0].strip()
        if output_text
        else ""
    )


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):
    job_input = job.get("input", {})

    try:
        prompt = job_input.get(
            "prompt",
            "Analyze the supplied remote-sensing image."
        )

        metadata = job_input.get("metadata", {})

        conversation_history = job_input.get(
            "conversation_history",
            []
        )

        task_type = detect_task_type(job_input)

        images, image_labels = collect_images(
            job_input
        )

        first_image = images[0]

        final_prompt = build_prompt(
            prompt=prompt,
            task_type=task_type,
            image=first_image,
            image_count=len(images),
            image_labels=image_labels,
            conversation_history=conversation_history,
            metadata=metadata,
        )

        response_text = run_inference(
            images=images,
            prompt=final_prompt,
        )

        if not response_text:
            response_text = (
                "RSCoVLM completed inference but returned "
                "an empty response."
            )

        # Always attempt to parse coordinate-form output.
        # This handles cases where task_type was classified
        # as VQA even though the model returned a box.
        groundings = parse_groundings(
            response_text,
            first_image.width,
            first_image.height,
        )

        result = {
            "response_text": response_text,
            "groundings": groundings,
            "task_type": task_type,
            "model_used": MODEL_ID,
            "metadata": {
                "image_count": len(images),
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

        print(
            "SNZ inference complete:",
            {
                "task_type": task_type,
                "groundings_count": len(groundings),
                "image_count": len(images),
            }
        )

        return result

    except Exception as exc:
        traceback.print_exc()

        return {
            "response_text": "",
            "groundings": [],
            "model_used": MODEL_ID,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# START WORKER
# ============================================================

if __name__ == "__main__":
    load_model()

    runpod.serverless.start({
        "handler": handler
    })
