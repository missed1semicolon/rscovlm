import os
import io
import re
import json
import base64
import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch
import runpod

from PIL import Image, ImageOps

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

# Qwen2.5-VL smart-resizes images to dimensions that are multiples
# of 28 pixels. The processor accepts the pixel limits below.
MIN_PIXELS = int(
    os.getenv("MIN_PIXELS", str(256 * 28 * 28))
)

MAX_PIXELS = int(
    os.getenv("MAX_PIXELS", str(1280 * 28 * 28))
)

# Qwen2.5-VL uses 14-pixel vision patches and a 2x2 spatial merge.
# image_grid_thw contains the PATCH grid dimensions, so each grid step
# represents 14 pixels in the resized image. The 28-pixel factor is used
# by smart_resize, but it is NOT the scale of image_grid_thw.
VISION_PATCH_SIZE = int(
    os.getenv("VISION_PATCH_SIZE", "14")
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

def load_model() -> None:
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

    image_b64 = image_b64.strip()

    # Support data URLs such as:
    # data:image/jpeg;base64,/9j/...
    if image_b64.startswith("data:image"):
        try:
            image_b64 = image_b64.split(",", 1)[1]
        except IndexError as exc:
            raise ValueError(
                "Invalid image data URL."
            ) from exc

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
        # Apply EXIF orientation before converting to RGB. This keeps
        # the image dimensions/geometry aligned with how a browser
        # normally displays an EXIF-oriented JPEG.
        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

    except Exception as exc:
        raise ValueError(
            f"Unable to decode image: {exc}"
        ) from exc

    return image


# ============================================================
# Image collection
# ============================================================

def collect_images(
    job_input: Dict[str, Any],
) -> Tuple[List[Image.Image], List[str]]:
    images: List[Image.Image] = []
    image_labels: List[str] = []

    image_b64 = job_input.get("image_b64")

    if image_b64:
        images.append(
            decode_base64_image(image_b64)
        )
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

            images.append(
                decode_base64_image(image_data)
            )
            image_labels.append(
                f"Image {index + 1}"
            )

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

            images.append(
                decode_base64_image(image_data)
            )
            image_labels.append(
                f"Additional image {index + 1}"
            )

    image_t1_b64 = job_input.get("image_t1_b64")
    image_t2_b64 = job_input.get("image_t2_b64")

    if image_t1_b64:
        images.append(
            decode_base64_image(image_t1_b64)
        )
        image_labels.append("Time 1 image")

    if image_t2_b64:
        images.append(
            decode_base64_image(image_t2_b64)
        )
        image_labels.append("Time 2 image")

    # ChangeFormer's predicted mask is included only when supplied.
    # For grounding, the PRIMARY image remains the coordinate target;
    # the mask is simply an additional visual input.
    mask_b64 = job_input.get("mask_b64")

    if mask_b64:
        images.append(
            decode_base64_image(mask_b64)
        )
        image_labels.append("Change-detection mask")

    if not images:
        raise ValueError(
            "No image was supplied. Provide image_b64, images_b64, "
            "additional_images_b64, image_t1_b64/image_t2_b64, "
            "or mask_b64."
        )

    return images, image_labels


# ============================================================
# SkySense++ fusion context
# ============================================================

def _format_fusion_evidence(
    fusion_context: Optional[Any],
) -> str:
    """
    Convert SkySense++ structured evidence into a compact prompt section.

    SkySense++ output is auxiliary model evidence, not ground truth.
    RSCoVLM must reconcile it against the actual supplied imagery.
    """
    if not fusion_context:
        return ""

    if not isinstance(fusion_context, dict):
        return (
            "\n\nSkySense++ auxiliary evidence:\n"
            "Evidence was supplied but was not in dictionary form. "
            "Treat it as unavailable."
        )

    model_name = (
        fusion_context.get("model")
        or "SkySense++"
    )

    evidence = fusion_context.get("evidence")

    if evidence is None:
        evidence = fusion_context.get(
            "skysense_evidence"
        )

    lines = [
        "\n\nSkySense++ multimodal auxiliary evidence:",
        f"Fusion specialist: {model_name}",
        (
            "This is model-generated auxiliary evidence, not ground truth. "
            "Verify it against the supplied optical and SAR imagery."
        ),
    ]

    if isinstance(evidence, dict):
        for key, value in evidence.items():
            if value is None:
                continue

            lines.append(
                f"- {key}: {value}"
            )

    elif evidence is not None:
        lines.append(
            f"- evidence: {evidence}"
        )

    modality_relationship = fusion_context.get(
        "modality_relationship"
    )

    if modality_relationship:
        lines.append(
            f"- modality relationship: {modality_relationship}"
        )

    analysis_mode = fusion_context.get(
        "analysis_mode"
    )

    if analysis_mode:
        lines.append(
            f"- analysis mode: {analysis_mode}"
        )

    return "\n".join(lines)


def _extract_fusion_overlay_b64(
    fusion_context: Optional[Any],
) -> Optional[str]:
    """Return an optional SkySense++ evidence-overlay Base64 string."""
    if not isinstance(fusion_context, dict):
        return None

    for key in (
        "evidence_overlay_b64",
        "skysense_evidence_overlay_b64",
        "overlay_b64",
    ):
        value = fusion_context.get(key)

        if isinstance(value, str) and value.strip():
            return value

    return None


def collect_fusion_overlay(
    fusion_context: Optional[Any],
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Decode an optional SkySense++ evidence overlay.

    The overlay is appended AFTER the primary image and therefore can
    never become the grounding target. Grounding coordinates always refer
    to images[0].
    """
    overlay_b64 = _extract_fusion_overlay_b64(
        fusion_context
    )

    if not overlay_b64:
        return None, None

    try:
        return (
            decode_base64_image(overlay_b64),
            "SkySense++ evidence overlay",
        )

    except Exception as exc:
        raise ValueError(
            f"Invalid SkySense++ evidence overlay: {exc}"
        ) from exc


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
    image_labels: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Any] = None,
    task_type: Optional[str] = None,
    fusion_context: Optional[Any] = None,
) -> str:
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
GROUNDING EXTRACTION MODE

The application needs machine-readable regions for the object or region
requested by the user. Inspect the full image carefully and identify ALL
visible instances that match the requested object or region.

MULTI-INSTANCE REQUIREMENTS:
- For requests using words such as "all", "every", or a plural object name,
  return a separate bounding box for each distinct visible instance/cluster.
- Do not return only the most prominent, highest-confidence, or nearest instance.
- Do not combine spatially separate instances into one large box.
- If multiple distinct instances are visible, include every instance that can
  be localized reliably. Do not invent instances or duplicate the same region.
- When previous grounding evidence is included in the conversation context,
  use it as a location cue, then verify each candidate against the current image.
  Include all matching visible regions, not only the previously highlighted one.

For this pass, output ONLY a JSON array. Do not output prose, explanations,
headings, markdown fences, or the user's question.

Each item must have exactly this shape:
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "object label"}
]

Coordinate contract:
- x1 = left edge
- y1 = top edge
- x2 = right edge
- y2 = bottom edge
- coordinates are absolute integer pixels in the image as presented to the
  vision model after visual preprocessing
- do NOT use normalized 0-1 coordinates, 0-1000 coordinates, percentages,
  or original-upload dimensions when the vision processor resized the image
- keep x1 <= x2 and y1 <= y2

If the requested object or region is not visibly identifiable, output exactly:
[]

Do not output the words "GROUNDING_JSON" or any natural-language answer in
this extraction pass.
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
- when SkySense++ auxiliary evidence is supplied, use it as a specialist cue
  and reconcile it against the actual optical and SAR imagery
- do not treat SkySense++ model output as ground truth
- do not invent geographic facts that cannot be supported by the supplied imagery

{grounding_instruction}

{
    "For this pass, follow the grounding extraction contract above and do not generate a user-facing answer."
    if grounding
    else "Give a concise, useful answer suitable for an analytical remote-sensing application."
}
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
        try:
            metadata_serialized = json.dumps(
                metadata,
                ensure_ascii=False,
                default=str,
            )
        except (TypeError, ValueError):
            metadata_serialized = str(metadata)

        metadata_text = (
            "\n\nImage metadata supplied by the application:\n"
            + metadata_serialized
        )

    history_text = ""

    if conversation_history:
        recent = conversation_history[-6:]
        history_lines: List[str] = []

        for item in recent:
            if not isinstance(item, dict):
                continue

            role = item.get("role", "user")
            content = item.get("content", "")

            if content:
                history_lines.append(
                    f"{role}: {content}"
                )

            # Include saved spatial evidence with the corresponding assistant
            # turn so follow-up grounding can resolve references such as
            # "the wildfires you mentioned" without relying on prose alone.
            prior_groundings = item.get("groundings")
            if role == "assistant" and isinstance(prior_groundings, list) and prior_groundings:
                compact_groundings = []
                for grounding_item in prior_groundings[:100]:
                    if not isinstance(grounding_item, dict):
                        continue
                    bbox = grounding_item.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        continue
                    compact_groundings.append({
                        "label": str(grounding_item.get("label") or "Detected Object"),
                        "bbox": [float(value) for value in bbox],
                        "coordinate_type": grounding_item.get(
                            "coordinate_type", "normalized_original_image"
                        ),
                        "coordinate_format": grounding_item.get(
                            "coordinate_format", "ymin,xmin,ymax,xmax"
                        ),
                    })
                if compact_groundings:
                    history_lines.append(
                        "previous assistant grounding evidence (JSON; normalized original-image "
                        "coordinates in ymin,xmin,ymax,xmax order): "
                        + json.dumps(compact_groundings, ensure_ascii=False)
                    )

        if history_lines:
            history_text = (
                "\n\nRecent conversation context:\n"
                + "\n".join(history_lines)
            )

    fusion_text = _format_fusion_evidence(
        fusion_context
    )

    return (
        system_instruction
        + labels_text
        + metadata_text
        + fusion_text
        + history_text
        + "\n\nUser question:\n"
        + prompt
    )


# ============================================================
# Qwen input construction
# ============================================================

def build_messages(
    images: List[Image.Image],
    prompt: str,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []

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


def prepare_inputs(
    messages: List[Dict[str, Any]],
):
    if processor is None:
        raise RuntimeError(
            "RSCoVLM processor has not been loaded."
        )

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

    return inputs, text


# ============================================================
# Qwen visual geometry
# ============================================================

def _grid_values_from_inputs(inputs) -> List[List[int]]:
    grid = inputs.get("image_grid_thw")

    if grid is None:
        raise RuntimeError(
            "Qwen processor did not return image_grid_thw. "
            "Grounding coordinates cannot be safely mapped back "
            "to the uploaded image."
        )

    if hasattr(grid, "detach"):
        grid_values = grid.detach().cpu().tolist()
    else:
        grid_values = grid.tolist()

    if not grid_values:
        raise RuntimeError(
            "Qwen processor returned an empty image_grid_thw."
        )

    normalized: List[List[int]] = []

    for row in grid_values:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            raise RuntimeError(
                "Unexpected image_grid_thw row: "
                f"{row!r}"
            )

        normalized.append(
            [
                int(row[0]),
                int(row[1]),
                int(row[2]),
            ]
        )

    return normalized


def get_model_view_sizes(
    inputs,
    image_count: int,
) -> List[Tuple[int, int]]:
    """
    Return one (width, height) pair for each supplied image.

    Qwen2.5-VL exposes image_grid_thw after preprocessing. Each row is
    [temporal, height_grid, width_grid]. The height/width entries are the
    vision PATCH grid, so each entry corresponds to processor.patch_size
    pixels (14 by default). The 2x2 merge changes tokenization, not the
    resized-image pixel dimensions.

    The first image's dimensions are used for the frontend grounding box.
    Supporting every image here prevents the old first-grid-entry-only
    assumption from silently becoming wrong if multiple images are sent.
    """
    if image_count <= 0:
        raise ValueError(
            "image_count must be greater than zero."
        )

    grid_values = _grid_values_from_inputs(inputs)

    if len(grid_values) < image_count:
        raise RuntimeError(
            "Qwen processor returned fewer image grids than supplied "
            f"images: grids={len(grid_values)}, images={image_count}."
        )

    sizes: List[Tuple[int, int]] = []

    for index in range(image_count):
        _, height_grid, width_grid = grid_values[index]

        model_height = (
            int(height_grid) * VISION_PATCH_SIZE
        )
        model_width = (
            int(width_grid) * VISION_PATCH_SIZE
        )

        if model_width <= 0 or model_height <= 0:
            raise RuntimeError(
                "Invalid Qwen model-view dimensions for image "
                f"{index + 1}: {model_width}x{model_height}."
            )

        sizes.append(
            (
                model_width,
                model_height,
            )
        )

    return sizes


# ============================================================
# Grounding parsing
# ============================================================

NUMBER = r"[-+]?\d+(?:\.\d+)?"

# Native RSCoVLM plain-text style:
#     x1,y1,x2,y2 object label
# Also accepts semicolons and optional brackets.
BBOX_LINE_RE = re.compile(
    rf"(?<!\d)"
    rf"\[?\s*({NUMBER})\s*[,;]\s*"
    rf"({NUMBER})\s*[,;]\s*"
    rf"({NUMBER})\s*[,;]\s*"
    rf"({NUMBER})\s*\]?"
    rf"\s*(?:[-–—:]\s*)?"
    rf"(.*)$",
    re.IGNORECASE,
)

# RSCoVLM/Qwen-style box markup sometimes appears as:
# <box>(x1,y1),(x2,y2)</box>
BOX_MARKUP_RE = re.compile(
    rf"<box>\s*\(\s*({NUMBER})\s*,\s*({NUMBER})\s*\)"
    rf"\s*,\s*\(\s*({NUMBER})\s*,\s*({NUMBER})\s*\)"
    rf"\s*</box>",
    re.IGNORECASE,
)

REF_MARKUP_RE = re.compile(
    r"<ref>\s*(.*?)\s*</ref>",
    re.IGNORECASE | re.DOTALL,
)


def clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    return max(
        low,
        min(high, value),
    )


def _clean_response_for_parsing(
    response_text: str,
) -> str:
    text = (response_text or "").strip()

    # Remove common markdown JSON fences without changing the underlying
    # coordinate values.
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def _json_candidates_from_text(
    text: str,
) -> List[Any]:
    """
    Find decodable JSON arrays/objects embedded in model output.

    This avoids the previous greedy:
        r"\\[\\s*\\{.*\\}\\s*\\]"
    which could swallow unrelated text when multiple bracketed regions
    were present.
    """
    candidates: List[Any] = []

    decoder = json.JSONDecoder()

    for match in re.finditer(
        r"[\[\{]",
        text,
    ):
        start = match.start()

        try:
            value, end = decoder.raw_decode(
                text[start:]
            )
        except json.JSONDecodeError:
            continue

        if end <= 0:
            continue

        candidates.append(value)

    return candidates


def _append_json_grounding_item(
    output: List[Dict[str, Any]],
    item: Dict[str, Any],
) -> None:
    bbox = (
        item.get("bbox_2d")
        or item.get("bbox")
        or item.get("box")
    )

    if not isinstance(
        bbox,
        (list, tuple),
    ) or len(bbox) != 4:
        return

    try:
        values = [
            float(value)
            for value in bbox
        ]
    except (TypeError, ValueError):
        return

    label = str(
        item.get("label")
        or item.get("sub_label")
        or item.get("name")
        or "Detected Object"
    ).strip()

    output.append(
        {
            "values": values,
            "label": label or "Detected Object",
            "raw_line": json.dumps(
                item,
                ensure_ascii=False,
            ),
        }
    )


def parse_grounding_lines(
    response_text: str,
) -> List[Dict[str, Any]]:
    """
    Extract grounding candidates from RSCoVLM output.

    Preferred format:
        [{"bbox_2d":[x1,y1,x2,y2],"label":"..."}]

    Fallback formats supported:
        x1,y1,x2,y2 object label
        <box>(x1,y1),(x2,y2)</box>
    """
    candidates: List[Dict[str, Any]] = []

    text = _clean_response_for_parsing(
        response_text
    )

    if not text:
        return candidates

    # --------------------------------------------------------
    # 1. Preferred JSON parsing
    # --------------------------------------------------------
    json_values = _json_candidates_from_text(text)

    for parsed in json_values:
        if isinstance(parsed, list):
            before = len(candidates)

            for item in parsed:
                if isinstance(item, dict):
                    _append_json_grounding_item(
                        candidates,
                        item,
                    )

            if len(candidates) > before:
                return candidates

        elif isinstance(parsed, dict):
            before = len(candidates)

            _append_json_grounding_item(
                candidates,
                parsed,
            )

            if len(candidates) > before:
                return candidates

    # --------------------------------------------------------
    # 2. <box> markup
    # --------------------------------------------------------
    ref_match = REF_MARKUP_RE.search(text)
    ref_label = (
        ref_match.group(1).strip()
        if ref_match
        else ""
    )

    for match in BOX_MARKUP_RE.finditer(text):
        values = [
            float(match.group(index))
            for index in range(1, 5)
        ]

        label = ref_label or "Detected Object"

        candidates.append(
            {
                "values": values,
                "label": label,
                "raw_line": match.group(0),
            }
        )

    if candidates:
        return candidates

    # --------------------------------------------------------
    # 3. Native RSCoVLM plain-text fallback
    # --------------------------------------------------------
    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        clean = re.sub(
            r"</?(?:box|ref|grounding)>",
            " ",
            line,
            flags=re.IGNORECASE,
        ).strip()

        match = BBOX_LINE_RE.search(clean)

        if not match:
            continue

        values = [
            float(match.group(index))
            for index in range(1, 5)
        ]

        label = match.group(5).strip(
            " -*_:`\t"
        )

        if not label:
            label = clean[
                :match.start()
            ].strip(
                " -*_:`\t"
            )

        if not label:
            label = "Detected Object"

        candidates.append(
            {
                "values": values,
                "label": label,
                "raw_line": line,
            }
        )

    return candidates


def extract_display_response(
    response_text: str,
    grounding: bool,
) -> str:
    """
    Keep the user-facing answer separate from the machine-readable
    grounding JSON while preserving the full raw model response for
    debugging.
    """
    text = (response_text or "").strip()

    if not text:
        return ""

    if not grounding:
        return text

    marker_match = re.search(
        r"(?im)^\s*GROUNDING_JSON\s*:\s*",
        text,
    )

    if marker_match:
        answer = text[:marker_match.start()].strip()

        if answer:
            answer = re.sub(
                r"(?im)^\s*(?:ANSWER|RESPONSE)\s*:\s*",
                "",
                answer,
                count=1,
            ).strip()

            return answer

    decoder = json.JSONDecoder()

    for match in re.finditer(r"[\[\{]", text):
        try:
            parsed, _ = decoder.raw_decode(
                text[match.start():]
            )
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, list):
            has_bbox = any(
                isinstance(item, dict)
                and (
                    "bbox_2d" in item
                    or "bbox" in item
                    or "box" in item
                )
                for item in parsed
            )
        elif isinstance(parsed, dict):
            has_bbox = (
                "bbox_2d" in parsed
                or "bbox" in parsed
                or "box" in parsed
            )
        else:
            has_bbox = False

        if has_bbox:
            answer = text[:match.start()].strip()

            if answer:
                answer = re.sub(
                    r"(?im)^\s*(?:ANSWER|RESPONSE)\s*:\s*",
                    "",
                    answer,
                    count=1,
                ).strip()

                return answer

            break

    # Never manufacture a natural-language answer from structured grounding
    # output. A grounding answer must come from a real model inference pass.
    return ""


# ============================================================
# Grounding coordinate conversion
# ============================================================

def convert_grounding_to_original(
    values: List[float],
    model_width: int,
    model_height: int,
    original_width: int,
    original_height: int,
) -> Tuple[List[float], str]:
    """
    Convert a Qwen2.5-VL / RSCoVLM model-view pixel box into the
    normalized coordinate system consumed by ImageBinder.

    Model output:
        [x1, y1, x2, y2]
        absolute pixels in the image presented to the vision model

    Frontend output:
        [ymin, xmin, ymax, xmax]
        normalized to [0, 1]

    Important:
    This function intentionally does NOT interpret values as 0-1000 or
    0-1 coordinates. RSCoVLM is based on Qwen2.5-VL, whose grounding
    coordinate convention is image-scale pixel coordinates.
    """
    if len(values) != 4:
        raise ValueError(
            "Expected four grounding coordinates, got: "
            f"{values}"
        )

    if (
        model_width <= 0
        or model_height <= 0
    ):
        raise ValueError(
            "Invalid model-view size: "
            f"{model_width}x{model_height}"
        )

    if (
        original_width <= 0
        or original_height <= 0
    ):
        raise ValueError(
            "Invalid original image size: "
            f"{original_width}x{original_height}"
        )

    try:
        x1, y1, x2, y2 = [
            float(value)
            for value in values
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric grounding coordinates: {values}"
        ) from exc

    if not all(
        math.isfinite(value)
        for value in (
            x1,
            y1,
            x2,
            y2,
        )
    ):
        raise ValueError(
            f"Non-finite grounding coordinates: {values}"
        )

    # Normalize corner ordering first.
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    # Do not silently turn a wildly out-of-range coordinate into a
    # misleading box. A small amount of overrun is allowed because
    # generative models can occasionally emit a coordinate one or two
    # pixels outside the visual boundary.
    tolerance_x = max(
        4.0,
        float(model_width) * 0.02,
    )
    tolerance_y = max(
        4.0,
        float(model_height) * 0.02,
    )

    if (
        left < -tolerance_x
        or right > float(model_width) + tolerance_x
        or top < -tolerance_y
        or bottom > float(model_height) + tolerance_y
    ):
        raise ValueError(
            "Grounding coordinates are outside the model-view image "
            f"range. box={values}, "
            f"model_view={model_width}x{model_height}"
        )

    # Clamp only the small allowed overrun.
    left = clamp(
        left,
        0.0,
        float(model_width),
    )
    right = clamp(
        right,
        0.0,
        float(model_width),
    )
    top = clamp(
        top,
        0.0,
        float(model_height),
    )
    bottom = clamp(
        bottom,
        0.0,
        float(model_height),
    )

    if (
        right <= left
        or bottom <= top
    ):
        raise ValueError(
            "Degenerate grounding box after validation: "
            f"[{left}, {top}, {right}, {bottom}]"
        )

    # Qwen's smart resize preserves the image aspect ratio, so mapping
    # model-view pixels back to original pixels is a scale operation.
    original_left = (
        left
        / float(model_width)
        * float(original_width)
    )

    original_right = (
        right
        / float(model_width)
        * float(original_width)
    )

    original_top = (
        top
        / float(model_height)
        * float(original_height)
    )

    original_bottom = (
        bottom
        / float(model_height)
        * float(original_height)
    )

    # ImageBinder expects normalized [ymin, xmin, ymax, xmax].
    ymin = clamp(
        original_top
        / float(original_height),
        0.0,
        1.0,
    )

    xmin = clamp(
        original_left
        / float(original_width),
        0.0,
        1.0,
    )

    ymax = clamp(
        original_bottom
        / float(original_height),
        0.0,
        1.0,
    )

    xmax = clamp(
        original_right
        / float(original_width),
        0.0,
        1.0,
    )

    return (
        [
            ymin,
            xmin,
            ymax,
            xmax,
        ],
        "model_view_pixels_xyxy",
    )


def parse_groundings(
    response_text: str,
    model_width: int,
    model_height: int,
    original_width: int,
    original_height: int,
) -> List[Dict[str, Any]]:
    groundings: List[Dict[str, Any]] = []

    candidates = parse_grounding_lines(
        response_text
    )

    for candidate in candidates:
        try:
            bbox, source_type = (
                convert_grounding_to_original(
                    candidate["values"],
                    model_width=model_width,
                    model_height=model_height,
                    original_width=original_width,
                    original_height=original_height,
                )
            )

        except ValueError as exc:
            print(
                "Skipping invalid grounding candidate:",
                candidate.get("raw_line"),
                "reason:",
                exc,
            )
            continue

        groundings.append(
            {
                "bbox": bbox,
                "label": candidate["label"],
                "confidence": None,

                # Frontend contract.
                "coordinate_type": (
                    "normalized_original_image"
                ),
                "coordinate_format": (
                    "ymin,xmin,ymax,xmax"
                ),

                # Original model output for debugging.
                "source_bbox": candidate["values"],
                "source_coordinate_type": source_type,
                "source_image_size": {
                    "width": model_width,
                    "height": model_height,
                },

                # Uploaded/display image dimensions.
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
def _generate_model_text(
    images: List[Image.Image],
    prompt: str,
    collect_confidence: bool = False,
) -> Tuple[str, Optional[float], List[Tuple[int, int]]]:
    """Run one genuine RSCoVLM generation pass."""
    messages = build_messages(
        images,
        prompt,
    )

    inputs, _ = prepare_inputs(
        messages
    )

    model_view_sizes = get_model_view_sizes(
        inputs,
        image_count=len(images),
    )

    inputs = inputs.to(
        model.device
    )

    generation = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        output_scores=collect_confidence,
        return_dict_in_generate=collect_confidence,
    )

    if collect_confidence:
        generated_ids = generation.sequences
    else:
        generated_ids = generation

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

    raw_response_text = (
        output_text[0].strip()
        if output_text
        else ""
    )

    confidence: Optional[float] = None

    if collect_confidence:
        try:
            transition_scores = model.compute_transition_scores(
                generation.sequences,
                generation.scores,
                normalize_logits=True,
            )

            token_ids = generated_ids_trimmed[0]
            token_scores = transition_scores[0]

            eos_token_id = getattr(
                processor.tokenizer,
                "eos_token_id",
                None,
            )

            pad_token_id = getattr(
                processor.tokenizer,
                "pad_token_id",
                None,
            )

            valid_scores = []

            for token_id, token_score in zip(
                token_ids,
                token_scores,
            ):
                token_id_int = int(
                    token_id.item()
                )

                if (
                    eos_token_id is not None
                    and token_id_int == eos_token_id
                ):
                    break

                if (
                    pad_token_id is not None
                    and token_id_int == pad_token_id
                ):
                    continue

                if torch.isfinite(token_score):
                    valid_scores.append(token_score)

            if valid_scores:
                mean_log_probability = torch.stack(
                    valid_scores
                ).mean()

                confidence = float(
                    torch.exp(
                        mean_log_probability
                    ).clamp(
                        min=0.0,
                        max=1.0,
                    ).item()
                )

        except Exception as exc:
            print(
                "Optional model-confidence calculation failed:",
                exc,
            )

    return (
        raw_response_text,
        confidence,
        model_view_sizes,
    )


def _build_grounding_answer_prompt(
    user_prompt: str,
    grounding_items: List[Dict[str, Any]],
) -> str:
    """Build the second, prose-only inference prompt for grounding."""
    grounding_evidence = json.dumps(
        [
            {
                "label": item.get("label", "object"),
                "bbox_2d": item.get("source_bbox", []),
            }
            for item in grounding_items
        ],
        ensure_ascii=False,
    )

    return f"""
You are RSCoVLM answering the user's remote-sensing question.

This is the FINAL ANSWER pass. Inspect the supplied image yourself and answer
based on the visible evidence. The preliminary grounding extraction below is
evidence to help you focus on the requested region; it is not a substitute for
visual reasoning and it is not guaranteed to be correct.

Application task and original user question:
{user_prompt}

Preliminary grounding evidence:
{grounding_evidence}

Write the actual natural-language answer to the user's question.

STRICT OUTPUT RULES:
- Output only the answer that should be shown to the user.
- Do not output JSON.
- Do not output bounding boxes or coordinates.
- Do not output the words GROUNDING_JSON, ANSWER:, RESPONSE:, or similar labels.
- Do not repeat or paraphrase these instructions.
- Never output the phrase "A concise natural-language answer."
- Do not invent details that are not supported by the image.
- If the evidence is uncertain, state the uncertainty naturally.
- Keep the answer concise but complete.
""".strip()


def _is_invalid_generated_answer(text: str) -> bool:
    """Reject structural/model-instruction text without inventing an answer."""
    value = (text or "").strip()
    if not value:
        return True

    lowered = value.lower().strip()

    if lowered in {
        "a concise natural-language answer.",
        "a concise natural-language answer",
        "a concise answer.",
        "a concise answer",
    }:
        return True

    if "grounding_json:" in lowered:
        return True

    if lowered.startswith("[") or lowered.startswith("{"):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return True
        except Exception:
            pass

    return False


@torch.inference_mode()
def run_inference(
    images: List[Image.Image],
    prompt: str,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    if model is None or processor is None:
        raise RuntimeError(
            "RSCoVLM worker is not initialized. "
            "load_model() must run before inference."
        )

    if not images:
        raise ValueError(
            "run_inference received no images."
        )

    grounding = is_grounding_task(
        task_type,
        prompt,
    )

    original_width = images[0].width
    original_height = images[0].height

    # --------------------------------------------------------
    # Normal VQA/captioning: one ordinary natural-language pass.
    # --------------------------------------------------------
    if not grounding:
        raw_response_text, confidence, model_view_sizes = (
            _generate_model_text(
                images=images,
                prompt=prompt,
                collect_confidence=True,
            )
        )

        return {
            "response_text": raw_response_text.strip(),
            "raw_response_text": raw_response_text,
            "groundings": [],
            "confidence": confidence,
            "confidence_method": (
                "geometric_mean_generated_token_probability"
                if confidence is not None
                else "unavailable"
            ),
            "original_image_size": {
                "width": original_width,
                "height": original_height,
            },
            "model_view_size": (
                {
                    "width": model_view_sizes[0][0],
                    "height": model_view_sizes[0][1],
                }
                if model_view_sizes
                else None
            ),
            "model_view_sizes": [
                {
                    "width": width,
                    "height": height,
                }
                for width, height in model_view_sizes
            ],
        }

    # --------------------------------------------------------
    # Grounding uses TWO genuine model inference passes:
    #   1. region extraction -> JSON only
    #   2. answer generation -> natural language only
    # This prevents the structured grounding contract from competing with
    # the user-facing prose contract in a single generation.
    # --------------------------------------------------------
    grounding_prompt = prompt

    raw_grounding_text, _, model_view_sizes = (
        _generate_model_text(
            images=images,
            prompt=grounding_prompt,
            collect_confidence=False,
        )
    )

    if not model_view_sizes:
        raise RuntimeError(
            "RSCoVLM did not return a model-view image size for grounding."
        )

    model_width, model_height = model_view_sizes[0]

    print(
        "Qwen visual model-view size "
        f"for grounding target: {model_width}x{model_height}"
    )

    if len(model_view_sizes) > 1:
        print(
            "Additional Qwen visual image sizes: "
            f"{model_view_sizes[1:]}"
        )

    groundings = parse_groundings(
        response_text=raw_grounding_text,
        model_width=model_width,
        model_height=model_height,
        original_width=original_width,
        original_height=original_height,
    )

    answer_prompt = _build_grounding_answer_prompt(
        user_prompt=prompt,
        grounding_items=groundings,
    )

    answer_text = ""
    answer_confidence: Optional[float] = None
    answer_raw_text = ""

    # Two answer attempts are still genuine RSCoVLM inference. The second
    # attempt is only used when the first generation violates the output
    # contract; no canned answer is ever substituted.
    for attempt in range(2):
        attempt_prompt = answer_prompt

        if attempt == 1:
            attempt_prompt = answer_prompt + "\n\nReturn only the actual answer text now."

        (
            candidate_text,
            candidate_confidence,
            _,
        ) = _generate_model_text(
            images=images,
            prompt=attempt_prompt,
            collect_confidence=True,
        )

        answer_raw_text = candidate_text
        answer_confidence = candidate_confidence

        if not _is_invalid_generated_answer(candidate_text):
            answer_text = candidate_text.strip()
            break

        print(
            "RSCoVLM grounding answer pass produced invalid structural "
            f"output on attempt {attempt + 1}; retrying with a stricter prompt."
        )

    return {
        "response_text": answer_text,
        "raw_response_text": answer_raw_text,
        "grounding_raw_response": raw_grounding_text,
        "groundings": groundings,
        "confidence": answer_confidence,
        "confidence_method": (
            "geometric_mean_generated_token_probability"
            if answer_confidence is not None
            else "unavailable"
        ),
        "original_image_size": {
            "width": original_width,
            "height": original_height,
        },
        "model_view_size": {
            "width": model_width,
            "height": model_height,
        },
        "model_view_sizes": [
            {
                "width": width,
                "height": height,
            }
            for width, height in model_view_sizes
        ],
    }


# ============================================================
# RunPod handler
# ============================================================

def handler(
    job: Dict[str, Any],
) -> Dict[str, Any]:
    job_input = job.get(
        "input",
        {},
    )

    if not isinstance(job_input, dict):
        return {
            "response_text": "",
            "groundings": [],
            "model_used": MODEL_ID,
            "confidence": None,
            "confidence_method": "unavailable",
            "error": (
                "RunPod job input must be an object/dictionary."
            ),
        }

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

        fusion_context = job_input.get(
            "fusion_context"
        )

        images, image_labels = collect_images(
            job_input
        )

        fusion_overlay, fusion_overlay_label = (
            collect_fusion_overlay(
                fusion_context
            )
        )

        if fusion_overlay is not None:
            images.append(fusion_overlay)
            image_labels.append(
                fusion_overlay_label
            )

        image_count = len(images)

        grounding = is_grounding_task(
            task_type,
            prompt,
        )

        print("=" * 70)
        print(
            "RSCoVLM JOB RECEIVED | "
            f"{image_count} IMAGE(S)"
        )
        print(
            f"Task type: {task_type}"
        )
        print(
            f"Grounding mode: {grounding}"
        )
        print(
            f"SkySense++ fusion: {bool(fusion_context)}"
        )
        print(
            f"SkySense++ overlay: {fusion_overlay is not None}"
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

        final_prompt = build_prompt(
            prompt=prompt,
            image_count=image_count,
            image_labels=image_labels,
            conversation_history=conversation_history,
            metadata=metadata,
            task_type=task_type,
            fusion_context=fusion_context,
        )

        inference = run_inference(
            images=images,
            prompt=final_prompt,
            task_type=task_type,
        )

        response_text = inference.get(
            "response_text",
            "",
        )

        # Never replace a failed/empty model answer with canned prose.
        # The response_text field must contain genuine RSCoVLM inference.

        result_metadata: Dict[str, Any] = {
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
            "original_image_size": inference.get(
                "original_image_size"
            ),
            "model_view_size": inference.get(
                "model_view_size"
            ),
            "model_view_sizes": inference.get(
                "model_view_sizes",
                [],
            ),
            "grounding_coordinate_contract": (
                "model_view_pixels_xyxy_to_"
                "normalized_original_ymin_xmin_ymax_xmax"
                if grounding
                else None
            ),
            "grounding_target_image_index": (
                0 if grounding else None
            ),
            "device": DEVICE,
            "confidence": inference.get(
                "confidence"
            ),
            "confidence_method": inference.get(
                "confidence_method",
                "unavailable",
            ),
            "fusion_context_received": bool(
                fusion_context
            ),
            "skysense_overlay_received": (
                fusion_overlay is not None
            ),
            "raw_model_response": inference.get(
                "raw_response_text",
                "",
            ),
        }

        result = {
            "response_text": response_text,
            "groundings": inference.get(
                "groundings",
                [],
            ),
            "model_used": MODEL_ID,
            "confidence": inference.get(
                "confidence"
            ),
            "confidence_method": inference.get(
                "confidence_method",
                "unavailable",
            ),
            "metadata": result_metadata,
        }

        if fusion_context:
            result["fusion_context"] = fusion_context

        return result

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
            "confidence": None,
            "confidence_method": "unavailable",
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
            "handler": handler,
        }
    )
