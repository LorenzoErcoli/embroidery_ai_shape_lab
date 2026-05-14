from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path

import numpy as np

from ai_plan_to_svg import (
    ellipse_bezier_path,
    fitted_ellipse_mask,
    keep_largest_components,
    parse_hex,
    smooth_mask,
)
from image_shape_lab import detect_subject_mask, load_rgb, mask_to_paths, remove_small_components


def bbox_to_pixels(bbox: dict, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, round(int(bbox["x1"]) / 1000 * width)))
    y1 = max(0, min(height - 1, round(int(bbox["y1"]) / 1000 * height)))
    x2 = max(0, min(width, round(int(bbox["x2"]) / 1000 * width)))
    y2 = max(0, min(height, round(int(bbox["y2"]) / 1000 * height)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def bbox_mask(shape: tuple[int, int], bbox: dict) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = bbox_to_pixels(bbox, w, h)
    result = np.zeros(shape, dtype=bool)
    result[y1:y2, x1:x2] = True
    return result


def color_mask(rgb: np.ndarray, color: tuple[int, int, int], tolerance: float) -> np.ndarray:
    target = np.array(color, dtype=np.float32)
    dist = np.linalg.norm(rgb.astype(np.float32) - target, axis=2)
    return dist <= tolerance


def mask_components(mask: np.ndarray) -> list[tuple[int, list[tuple[int, int]], tuple[int, int, int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            cells: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(sy, sx)])
            seen[sy, sx] = True
            min_x = max_x = sx
            min_y = max_y = sy
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            components.append((len(cells), cells, (min_x, min_y, max_x, max_y)))
    components.sort(key=lambda item: item[0], reverse=True)
    return components


def component_rect_paths(mask: np.ndarray, max_components: int) -> list[str]:
    paths = []
    for _, _, (min_x, min_y, max_x, max_y) in mask_components(mask)[:max_components]:
        paths.append(f"M {min_x} {min_y} L {max_x + 1} {min_y} L {max_x + 1} {max_y + 1} L {min_x} {max_y + 1} Z")
    return paths


def component_ellipse_paths(mask: np.ndarray, max_components: int) -> list[str]:
    paths = []
    for _, cells, _ in mask_components(mask)[:max_components]:
        comp = np.zeros_like(mask, dtype=bool)
        for y, x in cells:
            comp[y, x] = True
        _, stats = fitted_ellipse_mask(comp)
        path = ellipse_bezier_path(stats)
        if path:
            paths.append(path)
    return paths


def ellipse_stats_from_bbox(bbox: dict, width: int, height: int) -> dict:
    x1, y1, x2, y2 = bbox_to_pixels(bbox, width, height)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    rx = max(1, (x2 - x1) / 2)
    ry = max(1, (y2 - y1) / 2)
    return {
        "enabled": True,
        "bbox": [x1, y1, x2, y2],
        "center": [round(cx, 2), round(cy, 2)],
        "radius": [round(rx, 2), round(ry, 2)],
        "area": int(np.pi * rx * ry),
    }


def primitive_bars_paths(bbox: dict, width: int, height: int) -> list[str]:
    x1, y1, x2, y2 = bbox_to_pixels(bbox, width, height)
    bw = x2 - x1
    bh = y2 - y1
    bar_w = bw * 0.18
    gap = bw * 0.09
    baseline = y2 - bh * 0.12
    heights = [bh * 0.34, bh * 0.58, bh * 0.82]
    paths = []
    start = x1 + bw * 0.16
    slant = bw * 0.08
    for index, hgt in enumerate(heights):
        left = start + index * (bar_w + gap)
        right = left + bar_w
        top = baseline - hgt
        paths.append(
            "M "
            f"{left:.1f} {baseline:.1f} "
            f"L {right:.1f} {baseline:.1f} "
            f"L {right - slant:.1f} {top:.1f} "
            f"L {left - slant:.1f} {top:.1f} Z"
        )
    return paths


def layer_text(item: dict) -> str:
    return f"{item.get('name', '')} {item.get('notes', '')} {item.get('shape_hint', '')}".lower()


def is_logo_bars_layer(item: dict) -> bool:
    text = layer_text(item)
    return any(token in text for token in ("logo", "adidas", "stripe", "stripes", "striscia", "strisce"))


def primitive_rack_bars_paths(bbox: dict, width: int, height: int) -> list[str]:
    x1, y1, x2, y2 = bbox_to_pixels(bbox, width, height)
    bw = x2 - x1
    bh = y2 - y1
    thickness = max(4.0, min(bw, bh) * 0.06)
    y_top = y1 + bh * 0.28
    y_mid = y1 + bh * 0.44
    y_low = y1 + bh * 0.72
    x_left = x1 + bw * 0.12
    x_right = x1 + bw * 0.90
    return [
        thick_line_path((x_left, y_top), (x_right, y_top), thickness),
        thick_line_path((x_left + bw * 0.05, y_mid), (x_right - bw * 0.06, y_mid), thickness * 0.8),
        thick_line_path((x_left + bw * 0.12, y_top), (x_left + bw * 0.24, y_low), thickness * 0.8),
        thick_line_path((x_right - bw * 0.16, y_top), (x_right - bw * 0.32, y_low), thickness * 0.8),
    ]


def primitive_handlebar_paths(bbox: dict, width: int, height: int) -> list[str]:
    x1, y1, x2, y2 = bbox_to_pixels(bbox, width, height)
    bw = x2 - x1
    bh = y2 - y1
    thickness = max(5.0, min(bw, bh) * 0.08)
    stem_bottom = (x1 + bw * 0.24, y1 + bh * 0.88)
    stem_top = (x1 + bw * 0.48, y1 + bh * 0.38)
    grip = (x1 + bw * 0.86, y1 + bh * 0.28)
    return [
        thick_line_path(stem_bottom, stem_top, thickness),
        thick_line_path(stem_top, grip, thickness),
    ]


def primitive_ring_paths(bbox: dict, width: int, height: int) -> list[str]:
    stats = ellipse_stats_from_bbox(bbox, width, height)
    cx, cy = stats["center"]
    rx, ry = stats["radius"]
    outer = ellipse_bezier_path(stats)
    inner_stats = {
        "enabled": True,
        "center": [cx, cy],
        "radius": [rx * 0.55, ry * 0.55],
    }
    inner = ellipse_bezier_path(inner_stats)
    if not outer or not inner:
        return []
    paths = [outer + " " + inner]
    for i in range(5):
        angle = (i / 5) * 2 * np.pi - np.pi / 2
        px = cx + np.cos(angle) * rx * 0.38
        py = cy + np.sin(angle) * ry * 0.38
        size = min(rx, ry) * 0.12
        paths.append(
            f"M {px:.1f} {py - size:.1f} L {px + size * 0.35:.1f} {py - size * 0.25:.1f} "
            f"L {px + size:.1f} {py - size * 0.2:.1f} L {px + size * 0.45:.1f} {py + size * 0.2:.1f} "
            f"L {px + size * 0.6:.1f} {py + size:.1f} L {px:.1f} {py + size * 0.55:.1f} "
            f"L {px - size * 0.6:.1f} {py + size:.1f} L {px - size * 0.45:.1f} {py + size * 0.2:.1f} "
            f"L {px - size:.1f} {py - size * 0.2:.1f} L {px - size * 0.35:.1f} {py - size * 0.25:.1f} Z"
        )
    return paths


def primitive_wheel_ring_paths(bbox: dict, width: int, height: int) -> list[str]:
    stats = ellipse_stats_from_bbox(bbox, width, height)
    cx, cy = stats["center"]
    rx, ry = stats["radius"]
    outer = ellipse_bezier_path(stats)
    inner_stats = {
        "enabled": True,
        "center": [cx, cy],
        "radius": [rx * 0.68, ry * 0.68],
    }
    inner = ellipse_bezier_path(inner_stats)
    return [outer + " " + inner] if outer and inner else []


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


def primitive_tube_bars_paths(bbox: dict, width: int, height: int) -> list[str]:
    x1, y1, x2, y2 = bbox_to_pixels(bbox, width, height)
    bw = x2 - x1
    bh = y2 - y1
    p = {
        "rear": (x1 + bw * 0.12, y1 + bh * 0.78),
        "seat": (x1 + bw * 0.34, y1 + bh * 0.20),
        "crank": (x1 + bw * 0.45, y1 + bh * 0.72),
        "head": (x1 + bw * 0.82, y1 + bh * 0.28),
        "front": (x1 + bw * 0.88, y1 + bh * 0.78),
    }
    tubes = [
        (p["rear"], p["seat"]),
        (p["seat"], p["crank"]),
        (p["crank"], p["rear"]),
        (p["seat"], p["head"]),
        (p["head"], p["crank"]),
        (p["head"], p["front"]),
    ]
    return [thick_line_path(a, b, max(6.0, min(bw, bh) * 0.035)) for a, b in tubes]


def thick_line_path(a: tuple[float, float], b: tuple[float, float], width: float) -> str:
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    length = max(1.0, float(np.hypot(dx, dy)))
    nx = -dy / length * width / 2
    ny = dx / length * width / 2
    return (
        f"M {ax + nx:.1f} {ay + ny:.1f} "
        f"L {bx + nx:.1f} {by + ny:.1f} "
        f"L {bx - nx:.1f} {by - ny:.1f} "
        f"L {ax - nx:.1f} {ay - ny:.1f} Z"
    )


def looks_like_wheel_layer(item: dict, bbox: dict, width: int, height: int) -> bool:
    name = item.get("name", "").lower()
    x1, y1, x2, y2 = bbox_to_pixels(bbox, width, height)
    return ("ruota" in name or "wheel" in name) or ((x2 - x1) > width * 0.25 and (y2 - y1) > height * 0.25)


def looks_like_seed_layer(item: dict) -> bool:
    name = item.get("name", "").lower()
    notes = item.get("notes", "").lower()
    return "seed" in name or "semi" in name or "seme" in name or "cavità" in notes or "cavita" in notes


def structural_bars_paths(item: dict, width: int, height: int) -> list[str]:
    text = layer_text(item)
    if "telaio" in text or "frame" in text:
        return primitive_tube_bars_paths(item["bbox"], width, height)
    if "portapacchi" in text or "rack" in text:
        return primitive_rack_bars_paths(item["bbox"], width, height)
    if "manubrio" in text or "handlebar" in text:
        return primitive_handlebar_paths(item["bbox"], width, height)
    return []


def write_svg(path: Path, width: int, height: int, layers: list[dict], clip_path: str | None) -> None:
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    if clip_path:
        lines.append("  <defs>")
        lines.append('    <clipPath id="subject-clip">')
        lines.append(f'      <path d="{clip_path}"/>')
        lines.append("    </clipPath>")
        lines.append("  </defs>")
    for index, layer in enumerate(layers, start=1):
        color = layer["color"]
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", layer["name"]).strip("-") or f"layer-{index}"
        clip_attr = "" if layer["role"] == "base" or not clip_path else ' clip-path="url(#subject-clip)"'
        lines.append(
            f'  <g id="spatial-{index}-{name}" data-role="{layer["role"]}" '
            f'data-ai-name="{layer["name"]}" fill="#{color[0]:02x}{color[1]:02x}{color[2]:02x}" '
            f'stroke="none" fill-rule="evenodd"{clip_attr}>'
        )
        lines.append(f'    <title>{layer["notes"]}</title>')
        for d in layer["paths"]:
            lines.append(f'    <path d="{d}"/>')
        lines.append("  </g>")
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def layer_paths(mask: np.ndarray, hint: str, max_components: int, simplify: int) -> list[str]:
    if hint == "ellipse":
        return component_ellipse_paths(mask, max(1, max_components))
    if hint in {"bars", "tube_bars"}:
        return component_rect_paths(mask, max(1, max_components))
    cleaned = keep_largest_components(mask, max_components)
    return mask_to_paths(cleaned, simplify)


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte piano spaziale AI in SVG.")
    parser.add_argument("image")
    parser.add_argument("plan")
    parser.add_argument("--output", default="output")
    parser.add_argument("--color-tolerance", type=float, default=58.0)
    parser.add_argument("--max-size", type=int, default=900)
    parser.add_argument("--edge-smooth", type=float, default=1.2)
    parser.add_argument("--close-pixels", type=int, default=3)
    parser.add_argument("--simplify", type=int, default=7)
    parser.add_argument("--min-region-area", type=int, default=160)
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.output) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    image = load_rgb(image_path, args.max_size)
    rgb = np.array(image)
    h, w = rgb.shape[:2]

    subject_mask = detect_subject_mask(rgb, 35.0)
    subject_bbox_mask = bbox_mask((h, w), plan["subject"]["bbox"])
    subject_mask = subject_mask & subject_bbox_mask
    if plan["subject"]["shape_hint"] == "ellipse":
        subject_stats = ellipse_stats_from_bbox(plan["subject"]["bbox"], w, h)
        subject_mask = bbox_mask((h, w), plan["subject"]["bbox"])
        grid_y, grid_x = np.indices((h, w))
        cx, cy = subject_stats["center"]
        rx, ry = subject_stats["radius"]
        subject_mask = (((grid_x - cx) / rx) ** 2 + ((grid_y - cy) / ry) ** 2) <= 1.0
    else:
        subject_stats = {"enabled": False}
        subject_mask = smooth_mask(subject_mask, args.edge_smooth, args.close_pixels, args.min_region_area)
    subject_clip_path = ellipse_bezier_path(subject_stats) if subject_stats.get("enabled") else None

    rendered_layers: list[dict] = []
    used_overlay = np.zeros((h, w), dtype=bool)
    subject_base_used = False
    for item in sorted(plan["layers"], key=lambda layer: layer["priority"]):
        if not item["include"] or item["role"] == "discard":
            continue
        color = parse_hex(item["color_hex"])
        if color is None:
            continue
        region = bbox_mask((h, w), item["bbox"])
        hint = item["shape_hint"]
        max_components = max(1, int(item["max_components"]))
        x1, y1, x2, y2 = bbox_to_pixels(item["bbox"], w, h)
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        min_area = max(args.min_region_area, int(bbox_area * float(item["min_area_ratio"])))

        use_subject_base = item["role"] == "base" and not subject_base_used and hint == "ellipse"
        if use_subject_base and hint == "ellipse":
            path = subject_clip_path
            paths = [path] if path else mask_to_paths(subject_mask, args.simplify)
            mask = subject_mask
            subject_base_used = True
        elif use_subject_base:
            mask = subject_mask & region
            mask = smooth_mask(mask, args.edge_smooth, args.close_pixels, min_area)
            paths = layer_paths(mask, hint, max_components, args.simplify)
            subject_base_used = True
        else:
            structural_paths = structural_bars_paths(item, image.width, image.height) if hint == "bars" else []
            if hint == "tube_bars" or structural_paths:
                mask = region & subject_mask
                paths = structural_paths or primitive_tube_bars_paths(item["bbox"], image.width, image.height)
            elif hint == "bars" and is_logo_bars_layer(item):
                mask = region & subject_mask
                paths = primitive_bars_paths(item["bbox"], image.width, image.height)
            elif hint == "bars":
                mask = color_mask(rgb, color, args.color_tolerance) & subject_mask & region
                mask = smooth_mask(mask, args.edge_smooth, args.close_pixels, min_area)
                mask = remove_small_components(mask, min_area)
                paths = component_rect_paths(mask, max_components)
            elif hint == "wheel_rings" or (hint == "rings" and looks_like_wheel_layer(item, item["bbox"], image.width, image.height)):
                mask = region & subject_mask
                paths = primitive_wheel_ring_paths(item["bbox"], image.width, image.height)
            elif hint == "rings":
                mask = region & subject_mask
                paths = primitive_ring_paths(item["bbox"], image.width, image.height)
            elif hint == "seed_pattern" or looks_like_seed_layer(item):
                mask = color_mask(rgb, color, args.color_tolerance) & subject_mask & region
                mask = smooth_mask(mask, args.edge_smooth, args.close_pixels, min_area)
                mask = remove_small_components(mask, min_area)
                mask = keep_largest_components(mask, max_components)
                paths = component_ellipse_paths(mask, max_components)
            else:
                mask = color_mask(rgb, color, args.color_tolerance) & subject_mask & region
                mask = smooth_mask(mask, args.edge_smooth, args.close_pixels, min_area)
                mask = remove_small_components(mask, min_area)
                mask = keep_largest_components(mask, max_components)
                if item["role"] == "base" and hint == "smooth_blob":
                    mask = fill_holes(mask)
                    mask = keep_largest_components(mask, 1)
                if item["role"] not in {"detail", "outline"}:
                    mask = mask & ~used_overlay
                paths = layer_paths(mask, hint, max_components, args.simplify)

        if not paths:
            continue
        if item["role"] != "base":
            used_overlay |= mask
        rendered_layers.append(
            {
                "name": item["name"],
                "role": item["role"],
                "color": color,
                "notes": item["notes"],
                "paths": paths,
                "area": int(mask.sum()),
                "path_count": len(paths),
            }
        )

    write_svg(out_dir / "composition_spatial.svg", image.width, image.height, rendered_layers, subject_clip_path)
    report = {
        "input": str(image_path),
        "plan": str(Path(args.plan)),
        "subject": plan["subject"],
        "layers": [
            {
                "name": layer["name"],
                "role": layer["role"],
                "area": layer["area"],
                "path_count": layer["path_count"],
                "color_hex": "#{:02x}{:02x}{:02x}".format(*layer["color"]),
            }
            for layer in rendered_layers
        ],
    }
    (out_dir / "spatial_svg_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SVG spaziale salvato: {out_dir / 'composition_spatial.svg'}")


if __name__ == "__main__":
    main()
