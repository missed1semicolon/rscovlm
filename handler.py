import os
import io
import re
import json
import base64
import traceback
from typing import Any, Dict, List, Optional, Tuple

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
    "Qingyun/RSCoVLM-7B-2512",
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

# Qwen2.5-VL uses a 14-pixel patch with a spatial merge factor of 2,
# so one visual grid step corresponds to 28 image pixels.
VISION_GRID_SIZE = int(
    os.getenv("VISION_GRID_SIZE", "28")
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GROUNDING_TASK_NAMES = {
    "grounding",
    "detection",
    "bbox",
    "bounding_box",
}

GROUNDING_KEYWORDS = (
    "bounding box",
    "bounding boxes",
    "bbox",
    "locate",
    "localize",
    "localise",
    "where is",
    "where are",
    "find the",
    "find all",
    "detect",
    "highlight",
    "show me where",
)


# ============================================================
# Global model objects
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

    print(f"GPU count: {torch.cuda.device_count()}")

    for index in range(torch.cuda.device_count()):
        print(
            f"GPU {index}: "
            f"{torch.cuda.get_device_name(index)}"
        )

    print("-" * 70)
    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print("Processor loaded successfully.")

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
        raise ValueError("Image Base64 data is empty.")

    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(
            image_b64,
            validate=True,
        )
    except Exception as exc:
        raise ValueError(
            f"Invalid Base64 image data: {exc}"
        ) from exc

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Unable to decode image: {exc}"
        ) from exc

    return image


# ============================================================
# Image collection
# ============================================================

def collect_images(job_input: dict):
    images: List[Image.Image] = []
    image_labels: List[str] = []

    image_b64 = job_input.get("image_b64")

    if image_b64:
        images.append(decode_base64_image(image_b64))
        image_labels.append("Primary image")

    images_b64 = job_input.get("images_b64")

    if images_b64:
        if not isinstance(images_b64, list):
            raise ValueError(
                "'images_b64' must be a list of Base64 images."
            )

        for index, image_data in enumerate(images_b64):
            if not image_data:
                continue

            images.append(decode_base64_image(image_data))
            image_labels.append(f"Image {index + 1}")

    # The API's multi-image route uses this field.
    additional_images_b64 = job_input.get(
        "additional_images_b64"
    )

    if additional_images_b64:
        if not isinstance(additional_images_b64, list):
            raise ValueError(
                "'additional_images_b64' must be a list."
            )

        for index, image_data in enumerate(
            additional_images_b64
        ):
            if not image_data:
                continue

            images.append(decode_base64_image(image_data))
            image_labels.append(
                f"Additional image {index + 1}"
            )

    image_t1_b64 = job_input.get("image_t1_b64")
    image_t2_b64 = job_input.get("image_t2_b64")

    if image_t1_b64:
        images.append(decode_base64_image(image_t1_b64))
        image_labels.append("Time 1 image")

    if image_t2_b64:
        images.append(decode_base64_image(image_t2_b64))
        image_labels.append("Time 2 image")

    # ChangeFormer's predicted mask is deliberately included only when
    # present. It is an image in the RSCoVLM prompt, not a grounding target.
    mask_b64 = job_input.get("mask_b64")

    if mask_b64:
        images.append(decode_base64_image(mask_b64))
        image_labels.append("Change-detection mask")

    if not images:
        raise ValueError(
            "No image was supplied. Provide image_b64, images_b64, "
            "additional_images_b64, image_t1_b64/image_t2_b64, "
            "or mask_b64."
        )

    return images, image_labels


# ============================================================
# Task helpers
# ============================================================

def is_grounding_task(
    task_type: Optional[str],
    prompt: str,
) -> bool:
    task = (task_type or "").strip().lower()
    query = (prompt or "").strip().lower()

    if task in GROUNDING_TASK_NAMES:
        return True

    return any(
        keyword in query
        for keyword in GROUNDING_KEYWORDS
    )


# ============================================================
# Prompt construction
# ============================================================

def build_prompt(
    prompt: str,
    image_count: int,
    image_labels=None,
    conversation_history=None,
    metadata=None,
    task_type: Optional[str] = None,
):
    prompt = (prompt or "").strip()

    if not prompt:
        prompt = "Analyze the supplied remote sensing image."

    grounding = is_grounding_task(
        task_type,
        prompt,
    )

    if image_count == 1:
        image_instruction = """
You are analyzing one remote-sensing image.

Base your answer on the visible evidence in that image.
""".strip()
    else:
        image_instruction = f"""
You are analyzing {image_count} remote-sensing images.

Treat the supplied images as separate but related observations.

Compare them when the user's question requires comparison.

Do not assume that multiple images represent the same location or
same time unless the supplied metadata or user question indicates this.

When comparing images, explicitly distinguish:
- observations common to the images
- differences between the images
- uncertain interpretations
""".strip()

    grounding_instruction = ""

    if grounding:
        grounding_instruction = """

GROUNDING MODE

The user is asking you to locate an object or region in the image.
You MUST return a machine-readable bounding box for every object you
identify for the user's requested location.

Use this exact coordinate convention:

    x1, y1, x2, y2 - object label

where:
- x1 = left edge
- y1 = top edge
- x2 = right edge
- y2 = bottom edge

Coordinates are integer PIXELS in the image as presented to the vision
model after its visual preprocessing. Do NOT use the original uploaded
image dimensions. Do NOT use normalized 0-1 coordinates. Do NOT use the
0-1000 coordinate convention.

Return ONLY valid JSON for grounding results. Use this exact schema:
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "object label"}
]

The bbox is [left, top, right, bottom] in pixels. Do not add markdown or
prose before or after the JSON.
""".strip()

    system_instruction = f"""
You are RSCoVLM, a remote-sensing vision-language assistant.

{image_instruction}

Analyze the supplied remote-sensing imagery carefully.

Answer the user's question using information supported by the supplied imagery.

When appropriate:
- identify land-cover or scene characteristics
- describe visible objects
- describe spatial relationships
- identify built-up areas, roads, water, vegetation, agricultural regions,
  infrastructure, or other visible features
- compare supplied images when relevant
- distinguish observations from uncertain interpretations
- do not invent geographic facts that cannot be supported by the supplied imagery

{grounding_instruction}

Give a concise, useful answer suitable for an analytical remote-sensing application.
""".strip()

    labels_text = ""

    if image_labels:
        labels_text = (
            "\n\nImage ordering:\n"
            + "\n".join(
                f"{index + 1}. {label}"
                for index, label in enumerate(image_labels)
            )
        )

    metadata_text = ""

    if metadata:
        metadata_text = (
            "\n\nImage metadata supplied by the application:\n"
            f"{metadata}"
        )

    history_text = ""

    if conversation_history:
        recent = conversation_history[-6:]
        history_lines = []

        for item in recent:
            role = item.get("role", "user")
            content = item.get("content", "")

            if content:
                history_lines.append(
                    f"{role}: {content}"
                )

        if history_lines:
            history_text = (
                "\n\nRecent conversation context:\n"
                + "\n".join(history_lines)
            )

    return (
        system_instruction
        + labels_text
        + metadata_text
        + history_text
        + "\n\nUser question:\n"
        + prompt
    )


# ============================================================
# Qwen input construction
# ============================================================

def build_messages(images, prompt: str):
    content = []

    for image in images:
        content.append(
            {
                "type": "image",
                "image": image,
            }
        )

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def prepare_inputs(messages):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs, text


def get_model_view_size(inputs) -> Tuple[int, int]:
    """
    Return the actual spatial dimensions represented by Qwen's visual grid.

    Qwen2.5-VL exposes image_grid_thw after preprocessing. The first image
    has [temporal, height_grid, width_grid], and each spatial grid cell is
    28 image pixels for the Qwen2.5-VL processor configuration used here.
    """
    grid = inputs.get("image_grid_thw")

    if grid is None:
        raise RuntimeError(
            "Qwen processor did not return image_grid_thw; cannot safely "
            "map RSCoVLM grounding coordinates to the original image."
        )

    if hasattr(grid, "detach"):
        grid_values = grid.detach().cpu().tolist()
    else:
        grid_values = grid.tolist()

    if not grid_values:
        raise RuntimeError(
            "Qwen processor returned an empty image_grid_thw."
        )

    first = grid_values[0]

    if len(first) < 3:
        raise RuntimeError(
            f"Unexpected image_grid_thw value: {grid_values}"
        )

    height_grid = int(first[1])
    width_grid = int(first[2])

    model_height = height_grid * VISION_GRID_SIZE
    model_width = width_grid * VISION_GRID_SIZE

    if model_width <= 0 or model_height <= 0:
        raise RuntimeError(
            f"Invalid model-view dimensions: {model_width}x{model_height}"
        )

    return model_width, model_height


# ============================================================
# Grounding parser
# ============================================================

NUMBER = r"[-+]?\d+(?:\.\d+)?"

# Deliberately require four numbers followed by a separator/label. This
# avoids interpreting arbitrary four-number prose as a bounding box.
BBOX_LINE_RE = re.compile(
    rf"(?<!\d)\[?\s*({NUMBER})\s*[,;]\s*({NUMBER})\s*[,;]\s*"
    rf"({NUMBER})\s*[,;]\s*({NUMBER})\s*\]?\s*(?:[-–—:]\s*)?(.+)?$",
    re.IGNORECASE,
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_grounding_lines(response_text: str) -> List[Dict[str, Any]]:
    """Extract Qwen JSON grounding boxes, with a plain-text fallback."""
    candidates: List[Dict[str, Any]] = []
    text = (response_text or "").strip()

    # Preferred Qwen2.5-VL JSON format.
    json_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)

    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue

                    bbox = item.get("bbox_2d") or item.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        continue

                    try:
                        values = [float(value) for value in bbox]
                    except (TypeError, ValueError):
                        continue

                    label = str(
                        item.get("label")
                        or item.get("sub_label")
                        or "Detected Object"
                    ).strip()

                    candidates.append({
                        "values": values,
                        "label": label or "Detected Object",
                        "raw_line": json.dumps(item, ensure_ascii=False),
                    })

                if candidates:
                    return candidates
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Plain-text fallback: x1,y1,x2,y2 - object label
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        clean = re.sub(
            r"</?(?:box|ref|grounding)>",
            " ",
            line,
            flags=re.IGNORECASE,
        )

        match = re.search(
            rf"(?<!\d)\[?\s*({NUMBER})\s*[,;]\s*({NUMBER})\s*[,;]\s*"
            rf"({NUMBER})\s*[,;]\s*({NUMBER})\s*\]?",
            clean,
            re.IGNORECASE,
        )

        if not match:
            continue

        values = [
            float(match.group(index))
            for index in range(1, 5)
        ]

        label = clean[match.end():].strip(" -*_:`\t")
        if not label:
            label = clean[:match.start()].strip(" -*_:`\t")
        if not label:
            label = "Detected Object"

        candidates.append({
            "values": values,
            "label": label,
            "raw_line": line,
        })

    return candidates


def convert_grounding_to_original(
    values: List[float],
    model_width: int,
    model_height: int,
    original_width: int,
    original_height: int,
) -> Tuple[List[float], str]:
    """
    Convert an RSCoVLM/Qwen grounding box from the model-view pixel
    coordinate system into the normalized frontend coordinate system.

    Model output convention:
        [x1, y1, x2, y2]

    Frontend convention:
        [ymin, xmin, ymax, xmax]

    The model-view dimensions come from Qwen's image_grid_thw after the
    processor has resized the image. The original dimensions are the
    dimensions of the uploaded PIL image.
    """
    if len(values) != 4:
        raise ValueError(
            f"Expected four grounding coordinates, got: {values}"
        )

    if model_width <= 0 or model_height <= 0:
        raise ValueError(
            f"Invalid model-view size: {model_width}x{model_height}"
        )

    if original_width <= 0 or original_height <= 0:
        raise ValueError(
            f"Invalid original image size: "
            f"{original_width}x{original_height}"
        )

    x1, y1, x2, y2 = [float(value) for value in values]

    if not all(torch.isfinite(torch.tensor(value)).item() for value in (x1, y1, x2, y2)):
        raise ValueError(
            f"Non-finite grounding coordinates: {values}"
        )

    # Normalize coordinate ordering first. Models occasionally emit the
    # opposite corners in reverse order.
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    # Coordinates are pixels in the image actually presented to the model.
    # Clamp them before scaling so a slightly over-running model prediction
    # cannot produce invalid frontend coordinates.
    left = clamp(left, 0.0, float(model_width))
    right = clamp(right, 0.0, float(model_width))
    top = clamp(top, 0.0, float(model_height))
    bottom = clamp(bottom, 0.0, float(model_height))

    if right <= left or bottom <= top:
        raise ValueError(
            f"Degenerate grounding box after clamping: "
            f"[{left}, {top}, {right}, {bottom}]"
        )

    # Map model-view pixels back to the original image pixels.
    original_left = left / float(model_width) * float(original_width)
    original_right = right / float(model_width) * float(original_width)
    original_top = top / float(model_height) * float(original_height)
    original_bottom = bottom / float(model_height) * float(original_height)

    # ImageBinder expects normalized [ymin, xmin, ymax, xmax].
    ymin = clamp(original_top / float(original_height), 0.0, 1.0)
    xmin = clamp(original_left / float(original_width), 0.0, 1.0)
    ymax = clamp(original_bottom / float(original_height), 0.0, 1.0)
    xmax = clamp(original_right / float(original_width), 0.0, 1.0)

    return [ymin, xmin, ymax, xmax], "model_view_pixels"


def parse_groundings(
    response_text: str,
    model_width: int,
    model_height: int,
    original_width: int,
    original_height: int,
) -> List[Dict[str, Any]]:
    groundings = []

    for candidate in parse_grounding_lines(response_text):
        try:
            bbox, source_type = convert_grounding_to_original(
                candidate["values"],
                model_width=model_width,
                model_height=model_height,
                original_width=original_width,
                original_height=original_height,
            )
        except ValueError as exc:
            print(
                "Skipping invalid grounding line:",
                candidate["raw_line"],
                "reason:",
                exc,
            )
            continue

        groundings.append(
            {
                "bbox": bbox,
                "label": candidate["label"],
                "confidence": None,
                "coordinate_type": "normalized_original_image",
                "coordinate_format": "ymin,xmin,ymax,xmax",
                "source_bbox": candidate["values"],
                "source_coordinate_type": source_type,
                "source_image_size": {
                    "width": model_width,
                    "height": model_height,
                },
                "original_image_size": {
                    "width": original_width,
                    "height": original_height,
                },
                "raw_line": candidate["raw_line"],
            }
        )

    return groundings


# ============================================================
# RSCoVLM inference
# ============================================================

@torch.inference_mode()
def run_inference(
    images,
    prompt: str,
    task_type: Optional[str] = None,
):
    grounding = is_grounding_task(task_type, prompt)

    messages = build_messages(images, prompt)
    inputs, _ = prepare_inputs(messages)

    model_width, model_height = get_model_view_size(inputs)

    print(
        "Qwen visual model-view size: "
        f"{model_width}x{model_height}"
    )

    # The first image is the image whose coordinate space is exposed to the
    # frontend. For a normal single-image grounding request this is exactly
    # the uploaded image. Multi-image grounding is not requested by the
    # current agent, but the metadata makes the behavior explicit.
    original_width = images[0].width
    original_height = images[0].height

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
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    response_text = (
        output_text[0].strip()
        if output_text
        else ""
    )

    groundings = []

    if grounding and len(images) >= 1:
        groundings = parse_groundings(
            response_text=response_text,
            model_width=model_width,
            model_height=model_height,
            original_width=original_width,
            original_height=original_height,
        )

    return {
        "response_text": response_text,
        "groundings": groundings,
        "model_view_size": {
            "width": model_width,
            "height": model_height,
        },
        "original_image_size": {
            "width": original_width,
            "height": original_height,
        },
    }


# ============================================================
# RunPod handler
# ============================================================

def handler(job):
    job_input = job.get("input", {})

    try:
        prompt = job_input.get(
            "prompt",
            "Analyze the supplied remote sensing imagery.",
        )

        task_type = job_input.get(
            "task_type",
            "vqa",
        )

        metadata = job_input.get(
            "metadata",
            {},
        )

        conversation_history = job_input.get(
            "conversation_history",
            [],
        )

        images, image_labels = collect_images(job_input)
        image_count = len(images)

        grounding = is_grounding_task(
            task_type,
            prompt,
        )

        print("=" * 70)
        print(
            f"RSCoVLM JOB RECEIVED | {image_count} IMAGE(S)"
        )
        print(f"Task type: {task_type}")
        print(f"Grounding mode: {grounding}")
        print(f"Prompt: {prompt}")

        for index, image in enumerate(images):
            print(
                f"{image_labels[index]}: "
                f"{image.width}x{image.height}"
            )

        print("=" * 70)

        final_prompt = build_prompt(
            prompt=prompt,
            image_count=image_count,
            image_labels=image_labels,
            conversation_history=conversation_history,
            metadata=metadata,
            task_type=task_type,
        )

        inference = run_inference(
            images=images,
            prompt=final_prompt,
            task_type=task_type,
        )

        response_text = inference["response_text"]

        if not response_text:
            response_text = (
                "RSCoVLM completed inference "
                "but returned an empty response."
            )

        return {
            "response_text": response_text,
            "groundings": inference["groundings"],
            "model_used": MODEL_ID,
            "metadata": {
                "task_type": task_type,
                "grounding_enabled": grounding,
                "image_count": image_count,
                "images": [
                    {
                        "label": image_labels[index],
                        "width": image.width,
                        "height": image.height,
                    }
                    for index, image in enumerate(images)
                ],
                "original_image_size": inference[
                    "original_image_size"
                ],
                "model_view_size": inference[
                    "model_view_size"
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

    print("Starting RunPod serverless worker...")

    runpod.serverless.start(
        {
            "handler": handler
        }
    )
