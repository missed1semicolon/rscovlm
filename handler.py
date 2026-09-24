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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# ============================================================
# RSCoVLM V8 — SMART VISION REASONING WORKER
# ============================================================
# Supports:
# - VQA
# - caption/scene description
# - grounding
# - smart combined grounding + answer
# - change-mask interpretation
# - optical/SAR interpretation
# - object counts and spatial breakdowns
#
# Previous assistant responses are not treated as evidence. Conversation
# history is only included when the gateway explicitly marks it relevant.
# ============================================================

MODEL_ID = os.getenv("MODEL_ID", "Qingyun/RSCoVLM-7B-2512")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "768"))
MIN_PIXELS = int(os.getenv("MIN_PIXELS", str(256 * 28 * 28)))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(1280 * 28 * 28)))
VISION_PATCH_SIZE = int(os.getenv("VISION_PATCH_SIZE", "14"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GROUNDING_TASK_NAMES = {
    "grounding", "detection", "bbox", "bounding_box", "smart_grounding"
}
GROUNDING_KEYWORDS = (
    "bounding box", "bounding boxes", "bbox", "locate", "localize", "localise",
    "where is", "where are", "find the", "find all", "detect", "highlight", "show me where",
)

model = None
processor = None


# ============================================================
# Model loading / image decoding
# ============================================================

def load_model() -> None:
    global model, processor
    print("=" * 70)
    print("SNZ RSCoVLM V8 WORKER STARTING")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the RSCoVLM worker.")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("RSCoVLM V8 loaded successfully.")


def decode_base64_image(value: str) -> Image.Image:
    if not value:
        raise ValueError("Image Base64 data is empty.")
    value = value.strip()
    if value.startswith("data:image"):
        value = value.split(",", 1)[1]
    try:
        raw = base64.b64decode(value, validate=True)
        image = Image.open(io.BytesIO(raw))
        return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to decode image: {exc}") from exc


def collect_images(job_input: Dict[str, Any]) -> Tuple[List[Image.Image], List[str]]:
    images: List[Image.Image] = []
    labels: List[str] = []

    def append(value: Optional[str], label: str) -> None:
        if value:
            images.append(decode_base64_image(value))
            labels.append(label)

    custom_labels = job_input.get("image_labels")
    if isinstance(custom_labels, list):
        custom_labels = [str(x) for x in custom_labels]
    else:
        custom_labels = []

    append(job_input.get("image_b64"), custom_labels[0] if len(custom_labels) > 0 else "Primary image")

    additional = job_input.get("additional_images_b64")
    if additional:
        if not isinstance(additional, list):
            raise ValueError("additional_images_b64 must be a list.")
        for i, value in enumerate(additional):
            label = custom_labels[i + 1] if len(custom_labels) > i + 1 else f"Additional image {i + 1}"
            append(value, label)

    if not images:
        # Compatibility with direct bi-temporal payloads.
        append(job_input.get("image_t1_b64"), "Pre-event image")
        append(job_input.get("image_t2_b64"), "Post-event image")

    mask = job_input.get("mask_b64")
    if mask:
        append(mask, "Change-detection mask")

    if not images:
        raise ValueError("No image was supplied.")
    return images, labels


# ============================================================
# SkySense++ auxiliary evidence
# ============================================================

def _fusion_overlay(fusion_context: Optional[Any]) -> Optional[Image.Image]:
    if not isinstance(fusion_context, dict):
        return None
    value = fusion_context.get("evidence_overlay_b64") or fusion_context.get("skysense_evidence_overlay_b64") or fusion_context.get("overlay_b64")
    if not isinstance(value, str) or not value.strip():
        return None
    return decode_base64_image(value)


def _fusion_text(fusion_context: Optional[Any]) -> str:
    if not fusion_context:
        return ""
    try:
        data = fusion_context if isinstance(fusion_context, dict) else {"evidence": str(fusion_context)}
        return (
            "\nSkySense++ auxiliary evidence (NOT ground truth):\n"
            + json.dumps(data, ensure_ascii=False, default=str)[:10000]
            + "\nVerify all such evidence against the supplied optical/SAR imagery."
        )
    except Exception:
        return ""


# ============================================================
# Task helpers / prompts
# ============================================================

def is_grounding_task(task_type: Optional[str], prompt: str) -> bool:
    task = (task_type or "").strip().lower()
    if task in GROUNDING_TASK_NAMES:
        return True
    return any(k in (prompt or "").lower() for k in GROUNDING_KEYWORDS)


def build_prompt(
    prompt: str,
    plan: Optional[Dict[str, Any]],
    image_labels: List[str],
    metadata: Any,
    conversation_history: Optional[List[Dict[str, Any]]],
    fusion_context: Optional[Any],
) -> str:
    plan = plan if isinstance(plan, dict) else {}
    operations = plan.get("operations", [])
    evidence = plan.get("evidence_requirements", [])
    labels_text = "\n".join(f"{i+1}. {label}" for i, label in enumerate(image_labels))

    operation_rules = []
    if "describe" in operations:
        operation_rules.append("Include a useful scene/feature description where it helps answer the question.")
    if "compare" in operations:
        operation_rules.append("Compare observations explicitly and separate similarities from differences.")
    if "change_mask" in operations:
        operation_rules.append("Use the ChangeFormer mask as a focus cue only; it is not ground truth.")
    if "count" in operations:
        operation_rules.append("Give an explicit numeric count or qualified estimate; do not say only 'many' or 'several'.")
    if "spatial_breakdown" in operations:
        operation_rules.append("Describe spatial clustering/left-right/top-bottom relationships only when supported by the image orientation.")
    if "impact_assessment" in operations:
        operation_rules.append("Separate directly visible effects from inferred secondary impacts.")
    if "fusion" in operations:
        operation_rules.append("Use complementary optical and SAR evidence and explain what each modality contributes.")
    if "uncertainty_check" in operations:
        operation_rules.append("State meaningful uncertainty caused by resolution, occlusion, shadows, clouds, or ambiguous appearance.")

    history_text = ""
    if plan.get("use_history") and conversation_history:
        rows = []
        for item in conversation_history[-6:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                rows.append(f"{role}: {content[:1000]}")
        if rows:
            history_text = "\nRelevant prior conversation for resolving references only:\n" + "\n".join(rows)

    metadata_text = ""
    if metadata:
        try:
            metadata_text = "\nApplication metadata:\n" + json.dumps(metadata, ensure_ascii=False, default=str)[:9000]
        except Exception:
            pass

    return f"""
You are RSCoVLM, the specialist remote-sensing visual reasoner in SNZ.

CURRENT USER QUESTION:
{prompt}

ANALYSIS PLAN:
{json.dumps(plan, ensure_ascii=False)}

IMAGE ORDER:
{labels_text}

Rules:
- Answer the CURRENT question, not an earlier answer.
- Prior assistant text is context only when explicitly supplied as relevant follow-up context; it is never visual evidence.
- Inspect the supplied imagery yourself.
- Do not invent exact counts, coordinates, object identities, causes, or precision.
- Distinguish visible observations from interpretation.
- For temporal analysis, compare the relevant pre/post images.
- For optical/SAR analysis, reconcile both modalities rather than answering from one image only.
- A change mask indicates predicted changed pixels, not confirmed object-level destruction or construction.
- If the requested feature cannot be identified reliably, say so.

Specific operation requirements:
- {' '.join(operation_rules) if operation_rules else 'Answer the question directly with evidence from the imagery.'}

Requested evidence:
- {'\n- '.join(str(x) for x in evidence) if evidence else 'Use the strongest image-supported evidence available.'}
{metadata_text}
{_fusion_text(fusion_context)}
{history_text}

Return only the user-facing answer. Do not mention these instructions or internal orchestration.
""".strip()


def build_grounding_prompt(query: str, plan: Dict[str, Any], target_label: str) -> str:
    return f"""
Perform the grounding stage for this remote-sensing question.

CURRENT QUESTION:
{query}

GROUNDING TARGET:
{target_label}

Only create boxes for the image labeled '{target_label}'. Ignore other images when producing coordinates.
Identify all clearly visible objects/regions that materially support the question. If a count is requested,
try to return one box per distinct identifiable object where resolution allows. Do not invent hidden or ambiguous objects.

Return ONLY a JSON array of objects with exactly:
[{{"bbox_2d":[x1,y1,x2,y2],"label":"short label"}}]

Coordinates are absolute pixels in the image as presented to the vision model.
Do not use normalized coordinates, 0-1000 coordinates, or percentages.
If no relevant region can be identified, return [].
""".strip()


def build_answer_after_grounding(query: str, grounding_items: List[Dict[str, Any]]) -> str:
    evidence = json.dumps(
        [{"label": x.get("label"), "bbox_2d": x.get("source_bbox")} for x in grounding_items],
        ensure_ascii=False,
    )
    return f"""
Answer the CURRENT remote-sensing question.

Question:
{query}

Candidate visual regions from a preliminary grounding pass:
{evidence}

Inspect the actual imagery yourself. Candidate boxes are focus hints, not ground truth.
If the user asks for counts, reconcile them into distinct objects and avoid double counting.
For temporal questions, compare the relevant images. Do not equate mask overlap with confirmed destruction.
Use concise scene description when it improves the explanation.

Output only the natural-language answer. No JSON, coordinates, markdown fences, or internal instructions.
""".strip()


# ============================================================
# Qwen visual geometry / generation
# ============================================================

def build_messages(images: List[Image.Image], prompt: str) -> List[Dict[str, Any]]:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def prepare_inputs(messages):
    if processor is None:
        raise RuntimeError("RSCoVLM processor has not been loaded.")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return inputs


def _grid_values(inputs) -> List[List[int]]:
    grid = inputs.get("image_grid_thw")
    if grid is None:
        raise RuntimeError("Qwen processor did not return image_grid_thw.")
    values = grid.detach().cpu().tolist() if hasattr(grid, "detach") else grid.tolist()
    return [[int(r[0]), int(r[1]), int(r[2])] for r in values]


def get_model_view_sizes(inputs, image_count: int) -> List[Tuple[int, int]]:
    rows = _grid_values(inputs)
    if len(rows) < image_count:
        raise RuntimeError(f"Qwen returned {len(rows)} image grids for {image_count} images.")
    sizes = []
    for _, h, w in rows[:image_count]:
        sizes.append((int(w) * VISION_PATCH_SIZE, int(h) * VISION_PATCH_SIZE))
    return sizes


@torch.inference_mode()
def generate_text(images: List[Image.Image], prompt: str, collect_confidence: bool) -> Tuple[str, Optional[float], List[Tuple[int, int]]]:
    messages = build_messages(images, prompt)
    inputs = prepare_inputs(messages)
    sizes = get_model_view_sizes(inputs, len(images))
    inputs = inputs.to(model.device)
    generation = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        output_scores=collect_confidence,
        return_dict_in_generate=collect_confidence,
    )
    sequences = generation.sequences if collect_confidence else generation
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, sequences)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    text = decoded[0].strip() if decoded else ""
    confidence = None
    if collect_confidence:
        try:
            scores = model.compute_transition_scores(generation.sequences, generation.scores, normalize_logits=True)
            token_scores = scores[0]
            token_ids = trimmed[0]
            eos = getattr(processor.tokenizer, "eos_token_id", None)
            pad = getattr(processor.tokenizer, "pad_token_id", None)
            valid = []
            for tid, score in zip(token_ids, token_scores):
                tid = int(tid.item())
                if eos is not None and tid == eos:
                    break
                if pad is not None and tid == pad:
                    continue
                if torch.isfinite(score):
                    valid.append(score)
            if valid:
                confidence = float(torch.exp(torch.stack(valid).mean()).clamp(0, 1).item())
        except Exception as exc:
            print("Confidence calculation unavailable:", exc)
    return text, confidence, sizes


# ============================================================
# Grounding parsing / coordinate conversion
# ============================================================

NUMBER = r"[-+]?\d+(?:\.\d+)?"
BOX_MARKUP_RE = re.compile(rf"<box>\s*\(\s*({NUMBER})\s*,\s*({NUMBER})\s*\)\s*,\s*\(\s*({NUMBER})\s*,\s*({NUMBER})\s*\)\s*</box>", re.I)
BBOX_LINE_RE = re.compile(rf"\[?\s*({NUMBER})\s*[,;]\s*({NUMBER})\s*[,;]\s*({NUMBER})\s*[,;]\s*({NUMBER})\s*\]?\s*(?:[-–—:]\s*)?(.*)$")


def _json_candidates(text: str) -> List[Any]:
    decoder = json.JSONDecoder()
    values = []
    for match in re.finditer(r"[\[\{]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            values.append(value)
        except json.JSONDecodeError:
            continue
    return values


def parse_grounding_candidates(text: str) -> List[Dict[str, Any]]:
    text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.I).strip()
    output = []

    for value in _json_candidates(text):
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_2d") or item.get("bbox") or item.get("box")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                values = [float(x) for x in bbox]
            except Exception:
                continue
            output.append({"values": values, "label": str(item.get("label") or item.get("name") or "Detected object"), "raw_line": json.dumps(item, ensure_ascii=False)})
        if output:
            return output

    for match in BOX_MARKUP_RE.finditer(text):
        output.append({"values": [float(match.group(i)) for i in range(1, 5)], "label": "Detected object", "raw_line": match.group(0)})
    if output:
        return output

    for line in text.splitlines():
        match = BBOX_LINE_RE.search(line.strip())
        if not match:
            continue
        output.append({
            "values": [float(match.group(i)) for i in range(1, 5)],
            "label": match.group(5).strip(" -*_:`\t") or "Detected object",
            "raw_line": line.strip(),
        })
    return output


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def convert_box(values: List[float], model_width: int, model_height: int, original_width: int, original_height: int) -> List[float]:
    if len(values) != 4:
        raise ValueError("Expected four coordinates.")
    x1, y1, x2, y2 = [float(x) for x in values]
    if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
        raise ValueError("Non-finite coordinates.")
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    tx = max(4.0, model_width * 0.02)
    ty = max(4.0, model_height * 0.02)
    if left < -tx or right > model_width + tx or top < -ty or bottom > model_height + ty:
        raise ValueError(f"Box outside model view: {values} vs {model_width}x{model_height}")
    left, right = _clamp(left, 0, model_width), _clamp(right, 0, model_width)
    top, bottom = _clamp(top, 0, model_height), _clamp(bottom, 0, model_height)
    if right <= left or bottom <= top:
        raise ValueError("Degenerate box.")
    ymin = _clamp((top / model_height), 0, 1)
    xmin = _clamp((left / model_width), 0, 1)
    ymax = _clamp((bottom / model_height), 0, 1)
    xmax = _clamp((right / model_width), 0, 1)
    return [ymin, xmin, ymax, xmax]


def parse_groundings(text: str, model_size: Tuple[int, int], original_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    model_w, model_h = model_size
    original_w, original_h = original_size
    results = []
    for item in parse_grounding_candidates(text):
        try:
            bbox = convert_box(item["values"], model_w, model_h, original_w, original_h)
        except Exception as exc:
            print("Skipping invalid grounding:", exc)
            continue
        results.append({
            "bbox": bbox,
            "label": item["label"],
            "confidence": None,
            "coordinate_type": "normalized_original_image",
            "coordinate_format": "ymin,xmin,ymax,xmax",
            "source_bbox": item["values"],
            "source_coordinate_type": "model_view_pixels_xyxy",
            "source_image_size": {"width": model_w, "height": model_h},
            "original_image_size": {"width": original_w, "height": original_h},
            "raw_line": item["raw_line"],
        })
    return results


# ============================================================
# Inference orchestration inside worker
# ============================================================

@torch.inference_mode()
def run_inference(images: List[Image.Image], prompt: str, task_type: str, plan: Dict[str, Any], target_index: int = 0) -> Dict[str, Any]:
    if model is None or processor is None:
        raise RuntimeError("RSCoVLM worker is not initialized.")
    if not images:
        raise ValueError("No images supplied.")

    grounding = bool(plan.get("needs_grounding")) or is_grounding_task(task_type, prompt)
    original_target = images[target_index if 0 <= target_index < len(images) else 0]
    original_size = original_target.size

    # Grounding pass first when it improves the answer.
    groundings: List[Dict[str, Any]] = []
    grounding_raw = ""
    grounding_confidence = None
    sizes: List[Tuple[int, int]] = []

    if grounding:
        target_label = str(plan.get("grounding_target_label") or f"image {target_index + 1}")
        extraction_prompt = build_grounding_prompt(prompt, plan, target_label)
        grounding_raw, _, sizes = generate_text(images, extraction_prompt, collect_confidence=False)
        if target_index >= len(sizes):
            raise RuntimeError("Grounding target image index is outside the returned Qwen image grids.")
        groundings = parse_groundings(grounding_raw, sizes[target_index], original_size)

        answer_prompt = build_answer_after_grounding(prompt, groundings)
        # One clean final answer pass. If the model emits structural text, retry once.
        for attempt in range(2):
            candidate, confidence, _ = generate_text(images, answer_prompt + ("\nReturn only the actual answer." if attempt else ""), collect_confidence=True)
            cleaned = candidate.strip()
            if cleaned and not cleaned.lower().startswith("grounding_json") and not (cleaned.startswith("[") and cleaned.endswith("]")):
                return {
                    "response_text": cleaned,
                    "raw_response_text": candidate,
                    "grounding_raw_response": grounding_raw,
                    "groundings": groundings,
                    "confidence": confidence,
                    "confidence_method": "geometric_mean_generated_token_probability" if confidence is not None else "unavailable",
                    "original_image_size": {"width": original_size[0], "height": original_size[1]},
                    "model_view_sizes": [{"width": w, "height": h} for w, h in sizes],
                    "model_view_size": {"width": sizes[target_index][0], "height": sizes[target_index][1]},
                }

        return {
            "response_text": "",
            "raw_response_text": grounding_raw,
            "grounding_raw_response": grounding_raw,
            "groundings": groundings,
            "confidence": grounding_confidence,
            "confidence_method": "unavailable",
            "original_image_size": {"width": original_size[0], "height": original_size[1]},
            "model_view_sizes": [{"width": w, "height": h} for w, h in sizes],
            "model_view_size": {"width": sizes[target_index][0], "height": sizes[target_index][1]},
        }

    # Normal VQA/caption/change/fusion answer.
    response, confidence, sizes = generate_text(images, prompt, collect_confidence=True)
    return {
        "response_text": response.strip(),
        "raw_response_text": response,
        "grounding_raw_response": "",
        "groundings": [],
        "confidence": confidence,
        "confidence_method": "geometric_mean_generated_token_probability" if confidence is not None else "unavailable",
        "original_image_size": {"width": original_size[0], "height": original_size[1]},
        "model_view_sizes": [{"width": w, "height": h} for w, h in sizes],
        "model_view_size": {"width": sizes[0][0], "height": sizes[0][1]} if sizes else None,
    }


# ============================================================
# RunPod handler
# ============================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input", {})
    if not isinstance(job_input, dict):
        return {"response_text": "", "groundings": [], "model_used": MODEL_ID, "error": "RunPod input must be an object."}

    try:
        prompt = str(job_input.get("prompt") or "Analyze the supplied remote-sensing imagery.")
        task_type = str(job_input.get("task_type") or "vqa")
        plan = job_input.get("analysis_plan") or job_input.get("metadata", {}).get("analysis_plan") or {}
        if not isinstance(plan, dict):
            plan = {}
        history = job_input.get("conversation_history") if isinstance(job_input.get("conversation_history"), list) else []
        metadata = job_input.get("metadata") or {}
        fusion_context = job_input.get("fusion_context")

        images, labels = collect_images(job_input)
        overlay = _fusion_overlay(fusion_context)
        if overlay is not None:
            images.append(overlay)
            labels.append("SkySense++ evidence overlay")

        # The gateway orders the grounding target first for change/fusion workflows.
        target_index = int(job_input.get("grounding_target_image_index", 0) or 0)
        target_index = max(0, min(target_index, len(images) - 1))
        plan = dict(plan)
        plan["grounding_target_label"] = labels[target_index]

        final_prompt = build_prompt(prompt, plan, labels, metadata, history, fusion_context)

        print("=" * 70)
        print("RSCoVLM V8 JOB")
        print("Task:", task_type)
        print("Operations:", plan.get("operations"))
        print("Grounding:", plan.get("needs_grounding"))
        print("Images:", labels)
        print("=" * 70)

        inference = run_inference(images, final_prompt, task_type, plan, target_index)

        result = {
            "response_text": inference.get("response_text", ""),
            "groundings": inference.get("groundings", []),
            "model_used": MODEL_ID,
            "confidence": inference.get("confidence"),
            "confidence_method": inference.get("confidence_method", "unavailable"),
            "metadata": {
                "task_type": task_type,
                "analysis_plan": plan,
                "image_count": len(images),
                "image_labels": labels,
                "grounding_enabled": bool(plan.get("needs_grounding")),
                "grounding_target_image_index": target_index,
                "original_image_size": inference.get("original_image_size"),
                "model_view_sizes": inference.get("model_view_sizes", []),
                "raw_model_response": inference.get("raw_response_text", ""),
                "grounding_raw_response": inference.get("grounding_raw_response", ""),
            },
        }
        if fusion_context:
            result["fusion_context"] = fusion_context
        return result

    except Exception as exc:
        print("=" * 70)
        print("RSCoVLM V8 INFERENCE ERROR")
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


if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
