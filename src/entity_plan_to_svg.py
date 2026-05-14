from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path

import numpy as np

from ai_plan_to_svg import keep_largest_components, smooth_mask
from image_shape_lab import detect_subject_mask, load_rgb, mask_to_paths, remove_small_components, save_mask

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import torch
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    SamPredictor = None
    sam_model_registry = None


HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_hex(value: str) -> tuple[int, int, int] | None:
    match = HEX_RE.match(value.strip())
    if not match:
        return None
    raw = match.group(1)
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def bbox_to_pixels(bbox: dict, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width, round(width * bbox["x1"] / 1000)))
    y1 = max(0, min(height, round(height * bbox["y1"] / 1000)))
    x2 = max(0, min(width, round(width * bbox["x2"] / 1000)))
    y2 = max(0, min(height, round(height * bbox["y2"] / 1000)))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def bbox_mask(shape: tuple[int, int], bbox: dict) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = bbox_to_pixels(bbox, w, h)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def color_mask(rgb: np.ndarray, color: tuple[int, int, int], tolerance: float) -> np.ndarray:
    target = np.array(color, dtype=np.float32)
    dist = np.linalg.norm(rgb.astype(np.float32) - target, axis=2)
    return dist <= tolerance


def adaptive_color_seed(
    rgb: np.ndarray,
    color: tuple[int, int, int],
    domain: np.ndarray,
    tolerance: float,
    min_pixels: int = 30,
) -> np.ndarray:
    for tol in (tolerance, tolerance * 1.35, tolerance * 1.75):
        seed = color_mask(rgb, color, tol) & domain
        if int(seed.sum()) >= min_pixels:
            return seed
    return color_mask(rgb, color, tolerance * 2.1) & domain


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    inverse = ~mask
    exterior = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if inverse[0, x]:
            q.append((0, x))
        if inverse[h - 1, x]:
            q.append((h - 1, x))
    for y in range(h):
        if inverse[y, 0]:
            q.append((y, 0))
        if inverse[y, w - 1]:
            q.append((y, w - 1))
    while q:
        y, x = q.popleft()
        if exterior[y, x] or not inverse[y, x]:
            continue
        exterior[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not exterior[ny, nx]:
                q.append((ny, nx))
    return mask | (inverse & ~exterior)


def ellipse_path(cx: float, cy: float, rx: float, ry: float) -> str:
    k = 0.5522847498
    return (
        f"M {cx - rx:.1f} {cy:.1f} "
        f"C {cx - rx:.1f} {cy - k * ry:.1f} {cx - k * rx:.1f} {cy - ry:.1f} {cx:.1f} {cy - ry:.1f} "
        f"C {cx + k * rx:.1f} {cy - ry:.1f} {cx + rx:.1f} {cy - k * ry:.1f} {cx + rx:.1f} {cy:.1f} "
        f"C {cx + rx:.1f} {cy + k * ry:.1f} {cx + k * rx:.1f} {cy + ry:.1f} {cx:.1f} {cy + ry:.1f} "
        f"C {cx - k * rx:.1f} {cy + ry:.1f} {cx - rx:.1f} {cy + k * ry:.1f} {cx - rx:.1f} {cy:.1f} Z"
    )


def ellipse_ring_path(bbox: dict, width: int, height: int, inner_ratio: float) -> str:
    x1, y1, x2, y2 = bbox_to_pixels(bbox, width, height)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    rx = max(1.0, (x2 - x1) / 2)
    ry = max(1.0, (y2 - y1) / 2)
    outer = ellipse_path(cx, cy, rx, ry)
    inner = ellipse_path(cx, cy, rx * inner_ratio, ry * inner_ratio)
    return f"{outer} {inner}"


def ellipse_mask_from_bbox(shape: tuple[int, int], bbox: dict, inner_ratio: float | None = None) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = bbox_to_pixels(bbox, w, h)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    rx = max(1.0, (x2 - x1) / 2)
    ry = max(1.0, (y2 - y1) / 2)
    grid_y, grid_x = np.indices((h, w))
    outer = (((grid_x - cx) / rx) ** 2 + ((grid_y - cy) / ry) ** 2) <= 1.0
    if inner_ratio is None:
        return outer
    inner = (((grid_x - cx) / (rx * inner_ratio)) ** 2 + ((grid_y - cy) / (ry * inner_ratio)) ** 2) <= 1.0
    return outer & ~inner


def stitch_order(layer: dict) -> tuple[int, int]:
    text = f'{layer["id"]} {layer["name"]}'.lower()
    tokens = set(re.findall(r"[a-z0-9_]+", text))
    if "frame" in tokens or "telaio" in tokens:
        stage = 10
    elif "fork" in tokens or "forcella" in tokens:
        stage = 12
    elif "fender" in tokens or "fenders" in tokens or "parafanghi" in tokens or "parafango" in tokens:
        stage = 14
    elif ("chain_guard" in tokens or "carter" in tokens) and "trim" not in tokens and "bordino" not in tokens:
        stage = 16
    elif "tire" in tokens or "pneumatico" in tokens:
        stage = 20
    elif "rim" in tokens or "cerchio" in tokens or "fianco" in tokens:
        stage = 22
    elif "rack" in tokens or "portapacchi" in tokens:
        stage = 30
    elif "saddle" in tokens or "sella" in tokens:
        stage = 32
    elif tokens & {"handlebar", "manubrio", "grip", "grips", "impugnature"}:
        stage = 34
    elif tokens & {"crank", "pedivella", "pedale"}:
        stage = 36
    elif "hub" in tokens or "mozzo" in tokens:
        stage = 40
    elif tokens & {"trim", "bordino", "filetto"}:
        stage = 50
    else:
        stage = 25 if layer["role"] == "overlay" else 18
    return stage, layer["source_priority"]


def edge_settings(policy: str, base_smooth: float, base_close: int, base_min_area: int) -> tuple[float, int, int]:
    if policy == "solid":
        return base_smooth * 1.35, base_close + 3, base_min_area * 2
    if policy == "smooth":
        return base_smooth * 1.15, base_close + 2, int(base_min_area * 1.5)
    if policy == "fine":
        return max(0.2, base_smooth * 0.55), max(0, base_close - 1), max(20, int(base_min_area * 0.5))
    return base_smooth, base_close, base_min_area


def mask_bbox_area(mask: np.ndarray) -> int:
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return 0
    return int((xx.max() - xx.min() + 1) * (yy.max() - yy.min() + 1))


def mask_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            cells: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(sy, sx)])
            seen[sy, sx] = True
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            components.append(
                {
                    "cells": cells,
                    "area": len(cells),
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "center": (sum(xs) / len(xs), sum(ys) / len(ys)),
                }
            )
    components.sort(key=lambda item: item["area"], reverse=True)
    return components


def component_mask(shape: tuple[int, int], components: list[dict]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for comp in components:
        for y, x in comp["cells"]:
            result[y, x] = True
    return result


def select_entity_components(
    mask: np.ndarray,
    entity: dict,
    max_components: int,
    image_area: int,
    bbox_area: int,
    is_logical_parent: bool,
) -> tuple[np.ndarray, str | None]:
    components = mask_components(mask)
    if not components:
        return mask, None
    max_area = max(1, int(image_area * float(entity["max_area_ratio"])))
    max_bbox_area = max(1, int(bbox_area * 0.82))
    filtered: list[dict] = []
    rejected_large = 0
    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]
        comp_bbox_area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
        too_large = comp["area"] > max_area or comp_bbox_area > max_bbox_area
        if too_large and not is_logical_parent and entity["role"] == "detail":
            rejected_large += 1
            continue
        filtered.append(comp)
    if not filtered:
        if is_logical_parent or entity["role"] in {"base", "overlay"}:
            return component_mask(mask.shape, components[:max_components]), f"kept_large_components={rejected_large}"
        return np.zeros_like(mask), f"all components rejected as too large ({rejected_large})"
    filtered.sort(key=lambda item: item["area"], reverse=True)
    return component_mask(mask.shape, filtered[:max_components]), f"rejected_large_components={rejected_large}" if rejected_large else None


def clamp_large_entity(mask: np.ndarray, entity: dict, image_area: int) -> tuple[np.ndarray, str | None]:
    if entity["role"] in {"base", "overlay"}:
        return mask, None
    max_area = max(1, int(image_area * float(entity["max_area_ratio"])))
    bbox_limit = image_area * (0.35 if entity["role"] == "detail" else 0.65)
    if int(mask.sum()) <= max_area and mask_bbox_area(mask) <= bbox_limit:
        return mask, None
    return np.zeros_like(mask), "entity rejected: area or bbox too large"


def chaikin_closed(points: list[tuple[float, float]], iterations: int) -> list[tuple[float, float]]:
    if len(points) < 4 or iterations <= 0:
        return points
    result = points
    for _ in range(iterations):
        next_points: list[tuple[float, float]] = []
        count = len(result)
        for index in range(count):
            x1, y1 = result[index]
            x2, y2 = result[(index + 1) % count]
            next_points.append((x1 * 0.75 + x2 * 0.25, y1 * 0.75 + y2 * 0.25))
            next_points.append((x1 * 0.25 + x2 * 0.75, y1 * 0.25 + y2 * 0.75))
        result = next_points
    return result


def quadratic_closed_path(points: list[tuple[float, float]]) -> str:
    if len(points) < 3:
        return ""
    start_x = (points[0][0] + points[-1][0]) / 2
    start_y = (points[0][1] + points[-1][1]) / 2
    parts = [f"M {start_x:.1f} {start_y:.1f}"]
    for index, (cx, cy) in enumerate(points):
        nx, ny = points[(index + 1) % len(points)]
        end_x = (cx + nx) / 2
        end_y = (cy + ny) / 2
        parts.append(f"Q {cx:.1f} {cy:.1f} {end_x:.1f} {end_y:.1f}")
    parts.append("Z")
    return " ".join(parts)


def contour_paths(mask: np.ndarray, simplify: float, smooth_iterations: int, min_points: int = 4) -> list[str]:
    if cv2 is None:
        return mask_to_paths(mask, int(max(1, simplify)))
    source = (mask.astype(np.uint8) * 255)
    contours, hierarchy = cv2.findContours(source, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    paths: list[str] = []
    for contour, meta in zip(contours, hierarchy[0]):
        area = abs(cv2.contourArea(contour))
        if area < 8:
            continue
        epsilon = max(0.35, simplify)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = [(float(point[0][0]), float(point[0][1])) for point in approx]
        if len(pts) < min_points:
            continue
        if smooth_iterations > 0 and len(pts) >= 6:
            pts = chaikin_closed(pts, smooth_iterations)
            command = quadratic_closed_path(pts)
        else:
            command = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts) + " Z"
        paths.append(command)
    return paths


def trace_paths(mask: np.ndarray, simplify: int) -> list[str]:
    return mask_to_paths(mask, max(1, int(simplify)))


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    if cv2 is not None:
        kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


def grabcut_rect(rgb: np.ndarray, bbox: dict) -> np.ndarray | None:
    if cv2 is None:
        return None
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox_to_pixels(bbox, w, h)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)
    try:
        cv2.grabCut(rgb[:, :, ::-1].copy(), mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


def grabcut_mask(rgb: np.ndarray, entity: dict, domain: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
    if cv2 is None:
        return None
    h, w = domain.shape
    if int(seed.sum()) < 20 or int(domain.sum()) < 50:
        return None
    region = bbox_mask((h, w), entity["bbox"])
    init = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    init[domain & region] = cv2.GC_PR_BGD
    init[seed] = cv2.GC_PR_FGD
    sure = erode_mask(seed, 2)
    init[sure] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb[:, :, ::-1].copy(), init, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    refined = ((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD)) & domain & region
    if int(refined.sum()) < max(10, int(seed.sum() * 0.25)):
        return None
    return refined


def logical_parent_ids(entities: list[dict]) -> set[str]:
    children_by_parent: dict[str, list[dict]] = {}
    for item in entities:
        parent_id = item.get("parent_id")
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(item)
    result: set[str] = set()
    for entity in entities:
        text = f'{entity["name"]} {entity["notes"]}'.lower()
        names_container = any(token in text for token in ("silhouette", "complessiva", "gruppo", "cluster", "parent", "vincolo"))
        not_final_fill = "non" in text and ("campitura" in text or "finale" in text)
        has_base_children = any(child.get("role") == "base" for child in children_by_parent.get(entity["id"], []))
        object_container = entity.get("entity_type") == "object" and has_base_children
        part_base = entity.get("entity_type") == "part" and entity.get("role") == "base"
        if (names_container and not part_base) or not_final_fill or object_container:
            result.add(entity["id"])
    return result


def write_svg(path: Path, width: int, height: int, layers: list[dict]) -> None:
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    for index, layer in enumerate(layers, start=1):
        color = layer["color"]
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", layer["name"]).strip("-") or f"layer-{index}"
        lines.append(
            f'  <g id="entity-{index}-{name}" data-role="{layer["role"]}" data-entity-id="{layer["id"]}" '
            f'data-ai-name="{layer["name"]}" fill="#{color[0]:02x}{color[1]:02x}{color[2]:02x}" stroke="none" fill-rule="evenodd">'
        )
        lines.append(f'    <title>{layer["notes"]}</title>')
        for d in layer["paths"]:
            lines.append(f'    <path d="{d}"/>')
        lines.append("  </g>")
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def sam_box_masks(
    predictor: object | None,
    bbox: dict,
    width: int,
    height: int,
) -> list[np.ndarray]:
    if predictor is None:
        return []
    x1, y1, x2, y2 = bbox_to_pixels(bbox, width, height)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return []
    box = np.array([x1, y1, x2, y2], dtype=np.float32)
    try:
        masks, _, _ = predictor.predict(box=box, multimask_output=True)
    except Exception:
        return []
    return [mask.astype(bool) for mask in masks]


def choose_sam_mask(candidates: list[np.ndarray], domain: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray | None:
    best_score = 0.0
    best: np.ndarray | None = None
    for candidate in candidates:
        clipped = candidate & domain
        area = int(clipped.sum())
        if area < 20:
            continue
        if seed is not None and int(seed.sum()) > 0:
            overlap = int((clipped & seed).sum())
            score = overlap / max(1, area) + overlap / max(1, int(seed.sum()))
        else:
            score = area / max(1, int(domain.sum()))
        if score > best_score:
            best_score = score
            best = clipped
    return best


def build_entity_mask(
    entity: dict,
    rgb: np.ndarray,
    subject: np.ndarray,
    parent_mask: np.ndarray,
    tolerance: float,
    segmentation: str,
    sam_predictor: object | None,
) -> np.ndarray:
    h, w = subject.shape
    region = bbox_mask((h, w), entity["bbox"])
    source = entity["mask_source"]
    color = parse_hex(entity["color_hex"])
    domain = subject & parent_mask & region
    loose_domain = subject & region
    if source == "subject_region":
        if segmentation in {"auto", "sam"}:
            selected = choose_sam_mask(sam_box_masks(sam_predictor, entity["bbox"], w, h), domain)
            if selected is not None:
                return selected
        if segmentation in {"auto", "grabcut"}:
            refined = grabcut_rect(rgb, entity["bbox"])
            if refined is not None:
                candidate = refined & domain
                if int(candidate.sum()) >= 30:
                    return candidate
                loose_candidate = refined & loose_domain
                if int(loose_candidate.sum()) >= 30:
                    return loose_candidate
        return domain
    if source == "parent_region":
        return parent_mask & region
    if source == "color_range" and color is not None:
        seed = adaptive_color_seed(rgb, color, domain, tolerance)
        search_domain = domain
        if int(seed.sum()) < 30:
            search_domain = loose_domain
            seed = adaptive_color_seed(rgb, color, search_domain, tolerance)
        if segmentation in {"auto", "sam"}:
            selected = choose_sam_mask(sam_box_masks(sam_predictor, entity["bbox"], w, h), search_domain, seed)
            if selected is not None:
                return selected
        if segmentation in {"auto", "grabcut"}:
            refined = grabcut_mask(rgb, entity, search_domain, seed)
            if refined is not None:
                return refined
        return seed
    return np.zeros_like(subject)


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte piano entita' AI in SVG mask-first.")
    parser.add_argument("image")
    parser.add_argument("plan")
    parser.add_argument("--output", default="output")
    parser.add_argument("--color-tolerance", type=float, default=58.0)
    parser.add_argument("--max-size", type=int, default=900)
    parser.add_argument("--edge-smooth", type=float, default=1.2)
    parser.add_argument("--close-pixels", type=int, default=4)
    parser.add_argument("--simplify", type=int, default=8)
    parser.add_argument("--min-region-area", type=int, default=180)
    parser.add_argument("--segmentation", choices=["color", "grabcut", "sam", "auto"], default="auto")
    parser.add_argument("--sam-checkpoint", default="models/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam-model", default="vit_b")
    parser.add_argument("--vector-smooth", type=int, default=1, help="Passaggi di smoothing Chaikin sui contorni SVG.")
    parser.add_argument(
        "--trace-style",
        choices=["smooth", "pixel"],
        default="smooth",
        help="smooth usa curve pulite; pixel mantiene piu' fedelta' tipo image trace grezzo.",
    )
    parser.add_argument("--subtract-overlays", action="store_true", help="Sottrae overlay gia' usati nello stesso parent.")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.output) / image_path.stem
    masks_dir = out_dir / "entity_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    image = load_rgb(image_path, args.max_size)
    rgb = np.array(image)
    h, w = rgb.shape[:2]
    image_area = h * w
    sam_predictor = None
    if args.segmentation in {"sam", "auto"} and sam_model_registry is not None:
        checkpoint = Path(args.sam_checkpoint)
        if checkpoint.exists():
            sam = sam_model_registry[args.sam_model](checkpoint=str(checkpoint))
            sam.to(device="cpu")
            sam_predictor = SamPredictor(sam)
            sam_predictor.set_image(rgb)

    subject = detect_subject_mask(rgb, 35.0)
    subject = smooth_mask(subject, args.edge_smooth, args.close_pixels, args.min_region_area)
    save_mask(subject, masks_dir / "_subject.png")

    masks: dict[str, np.ndarray] = {"__root__": subject}
    rendered: list[dict] = []
    report_entities: list[dict] = []
    used_by_parent: dict[str, np.ndarray] = {}
    logical_ids = logical_parent_ids(plan["entities"])

    for entity in sorted(plan["entities"], key=lambda item: item["priority"]):
        entity_id = entity["id"]
        parent_id = entity["parent_id"] or "__root__"
        parent_mask = masks.get(parent_id, subject)
        if not entity["include"] or entity["role"] == "exclude" or entity["mask_source"] == "exclude":
            masks[entity_id] = np.zeros_like(subject)
            report_entities.append({"id": entity_id, "name": entity["name"], "included": False, "reason": "excluded by AI"})
            continue

        bbox = entity["bbox"]
        x1, y1, x2, y2 = bbox_to_pixels(bbox, w, h)
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        is_logical_parent = entity_id in logical_ids
        mask = build_entity_mask(entity, rgb, subject, parent_mask, args.color_tolerance, args.segmentation, sam_predictor)
        min_area = max(args.min_region_area, int(bbox_area * float(entity["min_area_ratio"])))
        smooth, close, min_area = edge_settings(entity["edge_policy"], args.edge_smooth, args.close_pixels, min_area)
        mask = smooth_mask(mask, smooth, close, min_area)
        mask = remove_small_components(mask, min_area)
        if entity["role"] == "base":
            mask = fill_holes(mask)
        max_components = max(1, int(entity["max_components"]))
        mask, component_rejection = select_entity_components(
            mask,
            entity,
            max_components,
            image_area,
            bbox_area,
            is_logical_parent,
        )

        if args.subtract_overlays and entity["role"] in {"overlay", "detail"} and not is_logical_parent:
            used = used_by_parent.setdefault(parent_id, np.zeros_like(subject))
            if entity["role"] == "overlay":
                mask = mask & ~used
                used_by_parent[parent_id] = used | mask

        mask, rejection = clamp_large_entity(mask, entity, image_area)
        rejection = "; ".join(part for part in (component_rejection, rejection) if part) or None
        if args.trace_style == "pixel":
            paths = trace_paths(mask, args.simplify)
        else:
            paths = contour_paths(mask, args.simplify, args.vector_smooth)
        masks[entity_id] = mask
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", entity_id).strip("_") or "entity"
        save_mask(mask, masks_dir / f"{safe_id}.png")
        report_entities.append(
            {
                "id": entity_id,
                "parent_id": parent_id,
                "name": entity["name"],
                "role": entity["role"],
                "mask_source": entity["mask_source"],
                "area": int(mask.sum()),
                "path_count": len(paths),
                "logical_parent": is_logical_parent,
                "rejection": rejection,
            }
        )
        color = parse_hex(entity["color_hex"]) or (0, 0, 0)
        if paths and entity_id not in logical_ids:
            rendered.append(
                {
                    "id": entity_id,
                    "name": entity["name"],
                    "role": entity["role"],
                    "color": color,
                    "notes": entity["notes"],
                    "paths": paths,
                    "source_priority": entity["priority"],
                }
            )

    svg_path = out_dir / "composition_entity.svg"
    rendered.sort(key=stitch_order)
    write_svg(svg_path, image.width, image.height, rendered)
    report = {
        "input": str(image_path),
        "plan": str(Path(args.plan)),
        "image_reading": plan.get("image_reading", ""),
        "entities": report_entities,
        "rendered_layers": len(rendered),
    }
    (out_dir / "entity_svg_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SVG entita' salvato: {svg_path}")


if __name__ == "__main__":
    main()
