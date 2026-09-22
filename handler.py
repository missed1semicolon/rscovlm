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
    print("Model:", MODEL_ID)
    print("Device:", DEVICE)
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
# BASE64 IMAGE DECODING
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
# MODEL-VIEW RESIZING
# ============================================================

# RSCoVLM uses scalable bounding boxes whose coordinate resolution
# follows the image resolution presented to the model. Keep the
# grounding coordinate system explicit by giving the model a bounded
# model-view image and then mapping its box back to the original image.
GROUNDING_MAX_SIDE = int(os.getenv("GROUNDING_MAX_SIDE", "1008"))
MIN_GROUNDING_SIDE = int(os.getenv("MIN_GROUNDING_SIDE", "224"))


def resize_for_grounding(image):
    original_width, original_height = image.size

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Image has invalid dimensions.")

    longest_side = max(original_width, original_height)

    scale = min(
        1.0,
        GROUNDING_MAX_SIDE / longest_side,
    )

    if longest_side < MIN_GROUNDING_SIDE:
        scale = min(
            GROUNDING_MAX_SIDE / longest_side,
            MIN_GROUNDING_SIDE / longest_side,
        )

    model_width = max(1, round(original_width * scale))
    model_height = max(1, round(original_height * scale))

    if (model_width, model_height) == image.size:
        return image.copy()

    return image.resize(
        (model_width, model_height),
        Image.Resampling.LANCZOS,
    )


def prepare_model_images(images, task_type):
    if task_type != "grounding":
        return images

    return [
        resize_for_grounding(image)
        for image in images
    ]


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_images(job_input):
    images = []
    labels = []

    # Prefer explicit temporal pair fields when present.
    image_t1 = job_input.get("image_t1_b64")
    image_t2 = job_input.get("image_t2_b64")

    if image_t1:
        images.append(decode_base64_image(image_t1))
        labels.append("Before image (Time 1)")

    if image_t2:
        images.append(decode_base64_image(image_t2))
        labels.append("After image (Time 2)")

    # ChangeFormer's mask is the third image.
    mask_b64 = job_input.get("mask_b64")

    if mask_b64:
        mask = decode_base64_image(mask_b64).convert("RGB")
        images.append(mask)
        labels.append(
            "ChangeFormer binary change mask "
            "(white=detected change, black=unchanged)"
        )

    # Existing compatibility fields, used only when no
    # explicit temporal pair was supplied.
    if not image_t1 and not image_t2:
        image_b64 = job_input.get("image_b64")

        if image_b64:
            images.append(decode_base64_image(image_b64))
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
                    labels.append(f"Image {index + 1}")

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
# TASK INSTRUCTIONS
# ============================================================

def task_instruction(task_type, image):
    width, height = image.size

    if task_type == "caption":
        return f"""
TASK: IMAGE CAPTIONING

Image dimensions: width={width}, height={height} pixels.

Write a concise caption describing visible remote-sensing features.
Do not invent geographic locations or unsupported details.
""".strip()

    if task_type == "grounding":
        return f"""
TASK: VISUAL GROUNDING

The image presented to you is exactly {width} pixels wide by {height} pixels high.

Identify the object requested by the user.

Return exactly one line when the object is found:
x1,y1,x2,y2 - object label

Coordinate convention:
- x increases left to right.
- y increases top to bottom.
- (x1,y1) is the top-left corner.
- (x2,y2) is the bottom-right corner.
- Coordinates are PIXELS IN THE IMAGE YOU ARE CURRENTLY SEEING.
- x must be between 0 and {width}; y must be between 0 and {height}.
- Do not use coordinates from the original upload if the image has been resized.
- Do not use normalized 0-1 or 0-1000 coordinates.
- Do not invent a box if the object cannot be located.
""".strip()

    return """
TASK: REMOTE-SENSING VISUAL QUESTION ANSWERING

Answer using visible image evidence.
Distinguish observations from uncertainty.
Do not invent geographic facts.
""".strip()


# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def build_prompt(
    prompt,
    task_type,
    image,
    image_labels=None,
    conversation_history=None,
    metadata=None,
    has_mask=False,
):
    prompt = (prompt or "").strip()

    if not prompt:
        prompt = "Analyze the supplied remote-sensing images."

    instruction = task_instruction(
        task_type,
        image
    )

    labels_text = ""

    if image_labels:
        labels_text = (
            "\n\nImages are supplied in this order:\n"
            + "\n".join(
                f"{i + 1}. {label}"
                for i, label in enumerate(image_labels)
            )
        )

    mask_text = ""

    if has_mask:
        mask_text = """

CHANGE MASK INTERPRETATION:
The supplied ChangeFormer mask is a binary prediction:
- White pixels (255) indicate pixels predicted as changed.
- Black pixels (0) indicate pixels predicted as unchanged.
- Treat the mask as model output, not ground truth.
- Compare the mask with the before and after images.
- If the mask appears inconsistent with the images, say so.
- Do not claim the mask identifies the type of change by itself.
"""

    metadata_text = ""

    if metadata:
        metadata_text = (
            "\n\nApplication image metadata:\n"
            + str(metadata)
        )

    history_text = ""

    if conversation_history:
        if not isinstance(conversation_history, list):
            raise ValueError(
                "'conversation_history' must be a list."
            )

        recent = conversation_history[-8:]
        lines = []

        for item in recent:
            if not isinstance(item, dict):
                continue

            role = item.get("role", "user")
            content = item.get("content", "")

            if role not in ("user", "assistant"):
                continue

            if content:
                lines.append(f"{role}: {content}")

        if lines:
            history_text = (
                "\n\nRecent conversation:\n"
                + "\n".join(lines)
            )

    return (
        "You are RSCoVLM, a remote-sensing vision-language assistant.\n\n"
        + instruction
        + labels_text
        + mask_text
        + metadata_text
        + history_text
        + "\n\nCurrent user request:\n"
        + prompt
    )


# ============================================================
# PARSE PIXEL BOXES
# ============================================================

def parse_groundings(
    response_text,
    model_width,
    model_height,
    original_width,
    original_height,
):
    """
    Parse RSCoVLM grounding output in model-view pixel coordinates,
    then convert it to normalized original-image coordinates.

    Frontend contract:
        bbox = [ymin, xmin, ymax, xmax], each in [0, 1]
    """
    groundings = []

    if not response_text:
        return groundings

    if (
        model_width <= 0
        or model_height <= 0
        or original_width <= 0
        or original_height <= 0
    ):
        return groundings

    pattern = re.compile(
        r"""
        ^\s*
        \(?\s*
        (-?\d+(?:\.\d+)?)\s*[,;]\s*
        (-?\d+(?:\.\d+)?)\s*[,;]\s*
        (-?\d+(?:\.\d+)?)\s*[,;]\s*
        (-?\d+(?:\.\d+)?)
        \s*\)?
        (?:\s*[-:|]\s*|\s+)
        (.*?)
        \s*$
        """,
        re.VERBOSE,
    )

    scale_x = original_width / model_width
    scale_y = original_height / model_height

    for line in response_text.splitlines():
        match = pattern.match(line.strip())

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
            math.isfinite(v)
            for v in (x1, y1, x2, y2)
        ):
            continue

        if (
            min(x1, y1, x2, y2) < 0
            or x1 > model_width
            or x2 > model_width
            or y1 > model_height
            or y2 > model_height
        ):
            continue

        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))

        if right <= left or bottom <= top:
            continue

        original_left = left * scale_x
        original_right = right * scale_x
        original_top = top * scale_y
        original_bottom = bottom * scale_y

        groundings.append({
            "bbox": [
                max(
                    0.0,
                    min(1.0, original_top / original_height),
                ),
                max(
                    0.0,
                    min(1.0, original_left / original_width),
                ),
                max(
                    0.0,
                    min(1.0, original_bottom / original_height),
                ),
                max(
                    0.0,
                    min(1.0, original_right / original_width),
                ),
            ],
            "label": (
                match.group(5).strip()
                or "Detected object"
            ),
            "coordinate_type": "normalized_original_image",
            "coordinate_format": "ymin,xmin,ymax,xmax",
            "source_bbox": [
                left,
                top,
                right,
                bottom,
            ],
            "source_coordinate_type": "model_view_pixels",
            "source_image_size": [
                model_width,
                model_height,
            ],
            "original_image_size": [
                original_width,
                original_height,
            ],
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
            "Analyze the supplied remote-sensing images."
        )

        metadata = job_input.get("metadata", {})

        conversation_history = job_input.get(
            "conversation_history",
            []
        )

        task_type = detect_task_type(job_input)

        images, image_labels = collect_images(job_input)

        # Keep the original image for UI coordinates and create an
        # explicit model-view image for grounding.
        original_first_image = images[0]

        has_mask = bool(job_input.get("mask_b64"))

        model_images = prepare_model_images(
            images,
            task_type,
        )

        model_first_image = model_images[0]

        final_prompt = build_prompt(
            prompt=prompt,
            task_type=task_type,
            image=model_first_image,
            image_labels=image_labels,
            conversation_history=conversation_history,
            metadata=metadata,
            has_mask=has_mask,
        )

        response_text = run_inference(
            images=model_images,
            prompt=final_prompt,
        )

        if not response_text:
            response_text = (
                "RSCoVLM completed inference but returned "
                "an empty response."
            )

        if task_type == "grounding":
            groundings = parse_groundings(
                response_text,
                model_first_image.width,
                model_first_image.height,
                original_first_image.width,
                original_first_image.height,
            )
        else:
            groundings = []

        print(
            "RSCoVLM MODEL VIEW:",
            {
                "width": model_first_image.width,
                "height": model_first_image.height,
            },
        )
        print("RAW MODEL RESPONSE:", repr(response_text))
        print("PARSED GROUNDINGS:", groundings)

        return {
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
                        "model_view_width": model_images[index].width,
                        "model_view_height": model_images[index].height,
                    }
                    for index, image in enumerate(images)
                ],
                "mask_included": has_mask,
                "device": DEVICE,
            },
        }

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
