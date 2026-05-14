from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from color_trace_svg import hex_color, mask_to_curve_paths, smooth_mask, write_svg
from image_shape_lab import detect_subject_mask, load_rgb, remove_small_components


def parse_hex(value: str) -> np.ndarray:
    raw = value.strip().lstrip("#")
    if len(raw) != 6:
        return np.array([0, 0, 0], dtype=np.uint8)
    return np.array([int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)], dtype=np.uint8)


def bbox_mask(shape: tuple[int, int], bbox: dict | None, scale_x: float, scale_y: float) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if not bbox:
        mask[:, :] = True
        return mask
    x1 = max(0, min(w, int(round(float(bbox.get("x1", 0)) * scale_x))))
    y1 = max(0, min(h, int(round(float(bbox.get("y1", 0)) * scale_y))))
    x2 = max(0, min(w, int(round(float(bbox.get("x2", w)) * scale_x))))
    y2 = max(0, min(h, int(round(float(bbox.get("y2", h)) * scale_y))))
    if x2 <= x1 or y2 <= y1:
        mask[:, :] = True
    else:
        mask[y1:y2, x1:x2] = True
    return mask


def color_distance_mask(rgb: np.ndarray, target_rgb: np.ndarray, tolerance: float) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target = cv2.cvtColor(target_rgb.reshape((1, 1, 3)), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
    delta = lab - target
    delta[:, :, 0] *= 0.7
    return np.linalg.norm(delta, axis=2) <= tolerance


def entity_sort_key(entity: dict) -> tuple[int, int]:
    role_rank = {"base": 0, "overlay": 1, "detail": 2, "outline": 3}
    return int(entity.get("priority", 100)), role_rank.get(str(entity.get("role", "overlay")), 1)


def entity_specificity(entity: dict) -> int:
    entity_type = str(entity.get("entity_type", "")).lower()
    if entity_type in {"part", "fill", "decoration", "text"}:
        return 3
    if entity_type in {"reflection", "shadow", "texture", "detail"}:
        return 2
    if entity_type == "object":
        return 1
    return 2


def build_entity_masks(args: argparse.Namespace, rgb: np.ndarray, plan: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    h, w = rgb.shape[:2]
    subject = detect_subject_mask(rgb, args.bg_tolerance)
    scale_x = w / 1000.0
    scale_y = h / 1000.0

    entities = {item["id"]: item for item in plan.get("entities", []) if item.get("id")}
    masks: dict[str, np.ndarray] = {}
    excluded = np.zeros((h, w), dtype=bool)

    for entity in sorted(entities.values(), key=entity_sort_key):
        box = bbox_mask((h, w), entity.get("bbox"), scale_x, scale_y)
        source = str(entity.get("mask_source", "color_range"))
        role = str(entity.get("role", "overlay"))
        include = bool(entity.get("include"))

        if source == "subject_region":
            mask = subject & box
        elif source == "parent_region":
            parent_id = entity.get("parent_id")
            mask = masks.get(parent_id, subject) & box
        elif source == "exclude":
            target = parse_hex(str(entity.get("color_hex", "#000000")))
            mask = color_distance_mask(rgb, target, args.exclude_tolerance) & box
        else:
            target = parse_hex(str(entity.get("color_hex", "#000000")))
            mask = color_distance_mask(rgb, target, args.color_tolerance) & box

        entity_type = str(entity.get("entity_type", "")).lower()
        if role == "exclude" and entity_type == "shadow":
            mask &= ~subject

        parent_id = entity.get("parent_id")
        if parent_id and parent_id in masks:
            mask &= masks[parent_id]
        elif role != "exclude":
            mask &= subject | box if source == "subject_region" else subject

        box_ratio = float(box.mean())
        is_global_background = entity_type == "background" or ("sfondo" in str(entity.get("name", "")).lower() and box_ratio > 0.75)

        if role == "exclude" or not include:
            if role == "exclude" and not is_global_background:
                excluded |= mask
            masks[entity["id"]] = mask
            continue

        min_area = max(1, int(float(entity.get("min_area_ratio", 0.0)) * h * w * args.ai_min_area_weight))
        max_components = int(entity.get("max_components", 0) or 0)
        mask = remove_small_components(mask, max(args.min_component_area, min_area))
        close = args.base_close if role == "base" else args.overlay_close
        mask = smooth_mask(mask, close, 0)

        if max_components > 0:
            mask = keep_largest_components(mask, max_components)
        masks[entity["id"]] = mask

    return masks, excluded


def keep_largest_components(mask: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0:
        return mask
    bitmap = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <= limit:
        return mask
    out = np.zeros_like(bitmap)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:limit]:
        cv2.drawContours(out, [contour], -1, 255, thickness=cv2.FILLED)
    return out > 0


def has_included_base_child(entity: dict, entities: list[dict]) -> bool:
    entity_id = entity.get("id")
    return any(
        item.get("parent_id") == entity_id
        and item.get("include")
        and item.get("role") == "base"
        and entity_specificity(item) > entity_specificity(entity)
        for item in entities
    )


def run(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    plan_path = Path(args.plan)
    out_dir = Path(args.output) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_rgb(image_path, args.max_size)
    rgb = np.array(image)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entities = [item for item in plan.get("entities", []) if item.get("id")]
    masks, excluded = build_entity_masks(args, rgb, plan)

    included = [item for item in entities if item.get("include") and item.get("role") in {"base", "overlay", "detail", "outline"}]
    base_entities = [item for item in included if item.get("role") == "base" and not has_included_base_child(item, entities)]
    overlay_entities = [item for item in included if item.get("role") != "base"]

    occupied_base = np.zeros(rgb.shape[:2], dtype=bool)
    layers: list[dict] = []

    for entity in sorted(base_entities, key=lambda item: (entity_specificity(item), int(item.get("priority", 100))), reverse=True):
        mask = masks.get(entity["id"], np.zeros(rgb.shape[:2], dtype=bool)).copy()
        mask &= ~excluded
        mask &= ~occupied_base
        if int(mask.sum()) < args.min_layer_area:
            continue
        occupied_base |= mask
        color = hex_color(parse_hex(str(entity.get("color_hex", "#000000"))))
        paths = mask_to_curve_paths(mask, args.min_path_area, args.base_simplify, args.curve_strength)
        if paths:
            layers.append(
                {
                    "id": entity["id"],
                    "role": "base",
                    "label": entity["id"],
                    "color": color,
                    "area": int(mask.sum()),
                    "paths": paths,
                }
            )

    layers = sorted(layers, key=lambda item: next((int(e.get("priority", 100)) for e in entities if e.get("id") == item["id"]), 100))

    for entity in sorted(overlay_entities, key=entity_sort_key):
        mask = masks.get(entity["id"], np.zeros(rgb.shape[:2], dtype=bool)).copy()
        mask &= ~excluded
        parent_id = entity.get("parent_id")
        if parent_id in masks:
            mask &= masks[parent_id]
        if entity.get("role") in {"overlay", "detail", "outline"} and occupied_base.any():
            mask &= occupied_base | masks.get(parent_id, occupied_base)
        if int(mask.sum()) < args.min_layer_area:
            continue
        paths = mask_to_curve_paths(mask, args.min_path_area, args.simplify, args.curve_strength)
        if not paths:
            continue
        layers.append(
            {
                "id": entity["id"],
                "role": str(entity.get("role", "overlay")),
                "label": entity["id"],
                "color": hex_color(parse_hex(str(entity.get("color_hex", "#000000")))),
                "area": int(mask.sum()),
                "paths": paths,
            }
        )

    write_svg(out_dir / "composition_ai_layered_trace.svg", image.width, image.height, layers, "#ffffff")
    report = {
        "input": str(image_path),
        "plan": str(plan_path),
        "output": str(out_dir / "composition_ai_layered_trace.svg"),
        "excluded_area": int(excluded.sum()),
        "layers": [
            {"id": layer["id"], "role": layer["role"], "color": layer["color"], "area": layer["area"], "path_count": len(layer["paths"])}
            for layer in layers
        ],
    }
    (out_dir / "ai_layered_trace_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"AI layered trace salvato: {out_dir / 'composition_ai_layered_trace.svg'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace colore guidato dal piano entity AI con basi esclusive.")
    parser.add_argument("image")
    parser.add_argument("plan")
    parser.add_argument("--output", default="output")
    parser.add_argument("--max-size", type=int, default=1100)
    parser.add_argument("--bg-tolerance", type=float, default=35.0)
    parser.add_argument("--color-tolerance", type=float, default=34.0)
    parser.add_argument("--exclude-tolerance", type=float, default=28.0)
    parser.add_argument("--min-component-area", type=int, default=24)
    parser.add_argument("--ai-min-area-weight", type=float, default=0.12)
    parser.add_argument("--min-layer-area", type=int, default=45)
    parser.add_argument("--min-path-area", type=int, default=35)
    parser.add_argument("--simplify", type=float, default=1.0)
    parser.add_argument("--base-simplify", type=float, default=1.3)
    parser.add_argument("--curve-strength", type=float, default=0.32)
    parser.add_argument("--base-close", type=int, default=2)
    parser.add_argument("--overlay-close", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
