from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from image_shape_lab import (
    detect_subject_mask,
    keep_largest_component,
    load_rgb,
    mask_to_paths,
    remove_small_components,
)


HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_hex(value: str) -> tuple[int, int, int] | None:
    match = HEX_RE.match(value.strip())
    if not match:
        return None
    raw = match.group(1)
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def color_mask(rgb: np.ndarray, subject: np.ndarray, color: tuple[int, int, int], tolerance: float) -> np.ndarray:
    target = np.array(color, dtype=np.float32)
    dist = np.linalg.norm(rgb.astype(np.float32) - target, axis=2)
    return subject & (dist <= tolerance)


def estimate_background_color(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    corners = np.array(
        [
            rgb[0, 0],
            rgb[0, w - 1],
            rgb[h - 1, 0],
            rgb[h - 1, w - 1],
        ],
        dtype=np.float32,
    )
    return np.median(corners, axis=0)


def requests_external_shadow_removal(plan: dict) -> bool:
    background = str(plan.get("background_handling", "")).lower()
    if ("ombra" in background or "shadow" in background) and (
        "sfondo" in background or "estern" in background or "sotto" in background or "external" in background
    ):
        return True

    for layer in plan.get("layers", []):
        if layer.get("role") != "discard":
            continue
        text = f'{layer.get("name", "")} {layer.get("shape_description", "")}'.lower()
        if "ombra" in text or "shadow" in text:
            return True
    return False


def is_ball_subject(plan: dict) -> bool:
    text = f'{plan.get("subject_type", "")} {plan.get("composition_strategy", "")}'.lower()
    return "pallone" in text or "ball" in text or "sfera" in text


def fitted_ellipse_mask(mask: np.ndarray, padding: int = 0) -> tuple[np.ndarray, dict]:
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return mask, {"enabled": False, "reason": "empty mask"}
    min_x = max(0, int(xx.min()) - padding)
    max_x = min(mask.shape[1] - 1, int(xx.max()) + padding)
    min_y = max(0, int(yy.min()) - padding)
    max_y = min(mask.shape[0] - 1, int(yy.max()) + padding)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    rx = max(1, (max_x - min_x) / 2)
    ry = max(1, (max_y - min_y) / 2)
    grid_y, grid_x = np.indices(mask.shape)
    ellipse = (((grid_x - cx) / rx) ** 2 + ((grid_y - cy) / ry) ** 2) <= 1.0
    return ellipse, {
        "enabled": True,
        "bbox": [min_x, min_y, max_x, max_y],
        "center": [round(cx, 2), round(cy, 2)],
        "radius": [round(rx, 2), round(ry, 2)],
        "area": int(ellipse.sum()),
    }


def ellipse_bezier_path(shape_stats: dict) -> str | None:
    if not shape_stats.get("enabled"):
        return None
    cx, cy = shape_stats["center"]
    rx, ry = shape_stats["radius"]
    k = 0.5522847498
    return (
        f"M {cx - rx:.1f} {cy:.1f} "
        f"C {cx - rx:.1f} {cy - k * ry:.1f} {cx - k * rx:.1f} {cy - ry:.1f} {cx:.1f} {cy - ry:.1f} "
        f"C {cx + k * rx:.1f} {cy - ry:.1f} {cx + rx:.1f} {cy - k * ry:.1f} {cx + rx:.1f} {cy:.1f} "
        f"C {cx + rx:.1f} {cy + k * ry:.1f} {cx + k * rx:.1f} {cy + ry:.1f} {cx:.1f} {cy + ry:.1f} "
        f"C {cx - k * rx:.1f} {cy + ry:.1f} {cx - rx:.1f} {cy + k * ry:.1f} {cx - rx:.1f} {cy:.1f} Z"
    )


def keep_largest_components(mask: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0:
        return mask
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
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
            components.append(cells)
    components.sort(key=len, reverse=True)
    result = np.zeros_like(mask, dtype=bool)
    for cells in components[:limit]:
        for y, x in cells:
            result[y, x] = True
    return result


def role_component_limit(role: str, overlay_limit: int, detail_limit: int, outline_limit: int) -> int:
    if role == "detail":
        return detail_limit
    if role == "outline":
        return outline_limit
    return overlay_limit


def remove_shadow_like_pixels(
    rgb: np.ndarray,
    subject: np.ndarray,
    bg_color: np.ndarray,
    enabled: bool,
    shadow_tolerance: float,
    max_saturation: float,
    min_luma_gap: float,
) -> tuple[np.ndarray, dict]:
    stats = {
        "enabled": enabled,
        "removed_pixels": 0,
        "reason": "",
    }
    if not enabled:
        stats["reason"] = "disabled"
        return subject, stats

    rgb_float = rgb.astype(np.float32)
    channel_range = rgb.max(axis=2).astype(np.float32) - rgb.min(axis=2).astype(np.float32)
    bg_dist = np.linalg.norm(rgb.astype(np.float32) - bg_color, axis=2)
    luma = rgb_float @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    bg_luma = float(bg_color @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    lower_half = np.indices(subject.shape)[0] > subject.shape[0] * 0.42
    darker_than_background = luma <= (bg_luma - min_luma_gap)
    candidate = (
        subject
        & lower_half
        & (channel_range <= max_saturation)
        & (bg_dist <= shadow_tolerance)
        & darker_than_background
    )
    exterior_shadow = np.zeros_like(subject, dtype=bool)
    h, w = subject.shape
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if candidate[0, x]:
            q.append((0, x))
        if candidate[h - 1, x]:
            q.append((h - 1, x))
    for y in range(h):
        if candidate[y, 0]:
            q.append((y, 0))
        if candidate[y, w - 1]:
            q.append((y, w - 1))

    while q:
        y, x = q.popleft()
        if exterior_shadow[y, x] or not candidate[y, x]:
            continue
        exterior_shadow[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not exterior_shadow[ny, nx]:
                q.append((ny, nx))

    shadow = exterior_shadow
    removed = int(shadow.sum())
    if removed == 0:
        stats["reason"] = "no shadow-like pixels matched"
        return subject, stats

    cleaned = subject & ~shadow
    cleaned = keep_largest_component(cleaned)
    stats["removed_pixels"] = removed
    stats["reason"] = "removed low-saturation pixels near background in lower subject area"
    return cleaned, stats


def smooth_mask(mask: np.ndarray, radius: float, close_pixels: int, min_area: int) -> np.ndarray:
    result = mask
    if close_pixels > 0:
        size = close_pixels * 2 + 1
        image = Image.fromarray((result.astype(np.uint8) * 255), mode="L")
        image = image.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
        result = np.array(image) >= 128
    if radius > 0:
        image = Image.fromarray((result.astype(np.uint8) * 255), mode="L")
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        result = np.array(image) >= 128
    result = remove_small_components(result, min_area)
    return result


def assign_subject_to_base_masks(
    rgb: np.ndarray,
    subject: np.ndarray,
    base_layers: list[dict],
) -> list[np.ndarray]:
    if len(base_layers) == 1:
        return [subject]

    colors = np.array([item["color"] for item in base_layers], dtype=np.float32)
    subject_pixels = rgb[subject].astype(np.float32)
    distances = np.sum((subject_pixels[:, None, :] - colors[None, :, :]) ** 2, axis=2)
    assignments = np.argmin(distances, axis=1)
    masks = [np.zeros(subject.shape, dtype=bool) for _ in base_layers]
    yy, xx = np.where(subject)
    for index, mask in enumerate(masks):
        selected = assignments == index
        mask[yy[selected], xx[selected]] = True
    return masks


def subtract_previous_overlays(mask: np.ndarray, used: np.ndarray, mode: str, role: str) -> np.ndarray:
    if mode == "allow":
        return mask
    if mode == "details-only" and role not in {"detail", "outline"}:
        return mask
    return mask & ~used


def write_ai_svg(
    path: Path,
    width: int,
    height: int,
    base_layers: list[dict],
    overlays: list[dict],
) -> None:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    ]
    for index, base in enumerate(base_layers, start=1):
        color = base["color"]
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", base["name"].strip()).strip("-") or f"base-{index}"
        lines.append(
            f'  <g id="ai-base-{index}-{name}" data-role="base" '
            f'data-ai-name="{base["name"]}" fill="#{color[0]:02x}{color[1]:02x}{color[2]:02x}" '
            'stroke="none">'
        )
        lines.append(f'    <title>{base["description"]}</title>')
        for d in base["paths"]:
            lines.append(f'    <path d="{d}"/>')
        lines.append("  </g>")

    for index, overlay in enumerate(overlays, start=1):
        color = overlay["color"]
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", overlay["name"].strip()).strip("-") or f"layer-{index}"
        lines.append(
            f'  <g id="ai-{index}-{name}" data-role="{overlay["role"]}" '
            f'data-ai-name="{overlay["name"]}" fill="#{color[0]:02x}{color[1]:02x}{color[2]:02x}" '
            'stroke="none">'
        )
        lines.append(f'    <title>{overlay["description"]}</title>')
        for d in overlay["paths"]:
            lines.append(f'    <path d="{d}"/>')
        lines.append("  </g>")
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte un piano AI ricamo in SVG usando maschere colore.")
    parser.add_argument("image", help="Percorso immagine input.")
    parser.add_argument("plan", help="Percorso ai_plan.json.")
    parser.add_argument("--output", default="output", help="Cartella output.")
    parser.add_argument("--bg-tolerance", type=float, default=35.0)
    parser.add_argument("--color-tolerance", type=float, default=52.0)
    parser.add_argument("--min-region-area", type=int, default=80)
    parser.add_argument("--max-size", type=int, default=900)
    parser.add_argument("--simplify", type=int, default=3)
    parser.add_argument("--edge-smooth", type=float, default=1.6, help="Raggio smoothing maschere prima del vettoriale.")
    parser.add_argument("--close-pixels", type=int, default=1, help="Chiude piccoli buchi/tagli nelle maschere.")
    parser.add_argument(
        "--overlap-mode",
        choices=["allow", "details-only", "none"],
        default="details-only",
        help="Controlla se i livelli sopra possono coprirsi.",
    )
    parser.add_argument(
        "--shadow-mode",
        choices=["auto", "force", "off"],
        default="auto",
        help="Rimuove ombre grigie vicine allo sfondo quando il piano AI le segnala.",
    )
    parser.add_argument(
        "--shape-prior",
        choices=["auto", "ball", "off"],
        default="auto",
        help="Usa una sagoma geometrica pulita quando il soggetto lo permette.",
    )
    parser.add_argument("--shadow-tolerance", type=float, default=95.0)
    parser.add_argument("--shadow-max-saturation", type=float, default=28.0)
    parser.add_argument("--shadow-min-luma-gap", type=float, default=14.0)
    parser.add_argument("--max-overlay-components", type=int, default=0)
    parser.add_argument("--max-detail-components", type=int, default=0)
    parser.add_argument("--max-outline-components", type=int, default=0)
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.output) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    image = load_rgb(image_path, args.max_size)
    rgb = np.array(image)
    bg_color = estimate_background_color(rgb)
    subject_raw = detect_subject_mask(rgb, args.bg_tolerance)

    included_layers = [
        layer for layer in plan.get("layers", []) if layer.get("include") and layer.get("role") != "discard"
    ]
    base_ai_layers = [layer for layer in included_layers if layer.get("role") == "base"]
    if not base_ai_layers:
        initial_subject = subject_raw
        mean = rgb[initial_subject].mean(axis=0)
        base_ai_layers = [
            {
                "name": "base soggetto",
                "color_hex": "#{:02x}{:02x}{:02x}".format(*(int(round(v)) for v in mean)),
                "shape_description": "Sagoma unica del soggetto.",
            }
        ]

    base_layer_specs: list[dict] = []
    shadow_enabled = args.shadow_mode == "force" or (
        args.shadow_mode == "auto" and requests_external_shadow_removal(plan)
    )
    subject, shadow_stats = remove_shadow_like_pixels(
        rgb,
        subject_raw,
        bg_color,
        shadow_enabled,
        args.shadow_tolerance,
        args.shadow_max_saturation,
        args.shadow_min_luma_gap,
    )
    shape_prior_enabled = args.shape_prior == "ball" or (args.shape_prior == "auto" and is_ball_subject(plan))
    shape_prior_stats = {"enabled": False, "reason": "disabled"}
    if shape_prior_enabled:
        subject, shape_prior_stats = fitted_ellipse_mask(subject_raw)
    subject = smooth_mask(subject, args.edge_smooth, args.close_pixels, args.min_region_area)

    if len(base_ai_layers) == 1:
        layer = base_ai_layers[0]
        color = parse_hex(layer.get("color_hex", "")) or tuple(int(round(v)) for v in rgb[subject].mean(axis=0))
        base_layer_specs.append(
            {
                "name": layer.get("name", "base soggetto"),
                "color": color,
                "description": layer.get("shape_description", "Sagoma unica del soggetto."),
            }
        )
    else:
        for layer in sorted(base_ai_layers, key=lambda item: item.get("priority", 999)):
            color = parse_hex(layer.get("color_hex", ""))
            if color is None:
                continue
            base_layer_specs.append(
                {
                    "name": layer.get("name", "base"),
                    "color": color,
                    "description": layer.get("shape_description", ""),
                }
            )

    base_masks = assign_subject_to_base_masks(rgb, subject, base_layer_specs)
    base_layers: list[dict] = []
    for index, (layer, mask) in enumerate(zip(base_layer_specs, base_masks)):
        mask = smooth_mask(mask, args.edge_smooth, args.close_pixels, args.min_region_area)
        ellipse_path = ellipse_bezier_path(shape_prior_stats) if index == 0 and len(base_layer_specs) == 1 else None
        paths = [ellipse_path] if ellipse_path else mask_to_paths(mask, args.simplify)
        if not paths:
            continue
        base_layers.append(
            {
                "name": layer["name"],
                "color": layer["color"],
                "description": layer["description"],
                "paths": paths,
                "area": int(mask.sum()),
            }
        )

    overlays: list[dict] = []
    used_color_masks: dict[tuple[int, int, int], np.ndarray] = {}
    used_overlay_pixels = np.zeros(subject.shape, dtype=bool)
    skipped_layers: list[dict] = []
    for layer in sorted(included_layers, key=lambda item: item.get("priority", 999)):
        if layer.get("role") == "base":
            continue
        color = parse_hex(layer.get("color_hex", ""))
        if color is None:
            continue
        mask = color_mask(rgb, subject, color, args.color_tolerance)
        mask = smooth_mask(mask, args.edge_smooth, args.close_pixels, args.min_region_area)
        mask = subtract_previous_overlays(mask, used_overlay_pixels, args.overlap_mode, layer.get("role", "overlay"))
        mask = remove_small_components(mask, args.min_region_area)
        role = layer.get("role", "overlay")
        limit = role_component_limit(
            role,
            args.max_overlay_components,
            args.max_detail_components,
            args.max_outline_components,
        )
        mask = keep_largest_components(mask, limit)
        paths = mask_to_paths(mask, args.simplify)
        if not paths:
            continue
        previous = used_color_masks.get(color)
        if previous is not None:
            overlap = int((previous & mask).sum())
            smaller = max(1, min(int(previous.sum()), int(mask.sum())))
            if overlap / smaller > 0.92:
                skipped_layers.append(
                    {
                        "name": layer.get("name", "layer"),
                        "role": layer.get("role", "overlay"),
                        "color_hex": "#{:02x}{:02x}{:02x}".format(*color),
                        "reason": "Stesso colore e stessa area di un livello gia' esportato; serve segmentazione AI con maschera, non solo matching colore.",
                    }
                )
                continue
        used_color_masks[color] = mask
        used_overlay_pixels |= mask
        overlays.append(
            {
                "name": layer.get("name", "layer"),
                "role": layer.get("role", "overlay"),
                "color": color,
                "description": layer.get("shape_description", ""),
                "paths": paths,
                "area": int(mask.sum()),
            }
        )

    if not base_layers:
        paths = mask_to_paths(subject, args.simplify)
        mean = rgb[subject].mean(axis=0)
        base_layers.append(
            {
                "name": "base soggetto",
                "color": tuple(int(round(v)) for v in mean),
                "description": "Fallback: sagoma unica del soggetto.",
                "paths": paths,
                "area": int(subject.sum()),
            }
        )

    write_ai_svg(out_dir / "composition_ai.svg", image.width, image.height, base_layers, overlays)
    report = {
        "input": str(image_path),
        "ai_plan": str(Path(args.plan)),
        "subject_type": plan.get("subject_type"),
        "composition_strategy": plan.get("composition_strategy"),
        "manipulators": {
            "edge_smooth": args.edge_smooth,
            "close_pixels": args.close_pixels,
            "overlap_mode": args.overlap_mode,
            "shadow_mode": args.shadow_mode,
            "shadow": shadow_stats,
            "shadow_min_luma_gap": args.shadow_min_luma_gap,
            "shape_prior": args.shape_prior,
            "shape_prior_result": shape_prior_stats,
            "max_overlay_components": args.max_overlay_components,
            "max_detail_components": args.max_detail_components,
            "max_outline_components": args.max_outline_components,
            "subject_area_raw": int(subject_raw.sum()),
            "subject_area_final": int(subject.sum()),
        },
        "bases": [
            {
                "name": item["name"],
                "color_hex": "#{:02x}{:02x}{:02x}".format(*item["color"]),
                "area": item["area"],
                "path_count": len(item["paths"]),
            }
            for item in base_layers
        ],
        "overlays": [
            {
                "name": item["name"],
                "role": item["role"],
                "color_hex": "#{:02x}{:02x}{:02x}".format(*item["color"]),
                "area": item["area"],
                "path_count": len(item["paths"]),
            }
            for item in overlays
        ],
        "skipped_layers": skipped_layers,
        "risks": plan.get("risks", []),
        "discard_rules": plan.get("discard_rules", []),
    }
    (out_dir / "ai_svg_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SVG AI salvato: {out_dir / 'composition_ai.svg'}")


if __name__ == "__main__":
    main()
