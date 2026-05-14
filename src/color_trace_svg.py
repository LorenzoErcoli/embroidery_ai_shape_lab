from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from image_shape_lab import detect_subject_mask, load_rgb, remove_small_components


def hex_color(rgb: np.ndarray) -> str:
    r, g, b = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def fmt(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


def smooth_mask(mask: np.ndarray, close: int, open_size: int) -> np.ndarray:
    out = mask.astype(np.uint8) * 255
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close * 2 + 1, close * 2 + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if open_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size * 2 + 1, open_size * 2 + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    return out > 0


def choose_work_mask(rgb: np.ndarray, bg_tolerance: float, mode: str) -> tuple[np.ndarray, str]:
    if mode == "full":
        return np.ones(rgb.shape[:2], dtype=bool), "full"
    subject = detect_subject_mask(rgb, bg_tolerance)
    ratio = float(subject.mean())
    if mode == "subject":
        return subject, "subject"
    if 0.05 <= ratio <= 0.92:
        return subject, "subject-auto"
    return np.ones(rgb.shape[:2], dtype=bool), "full-auto"


def quantize_lab(rgb: np.ndarray, mask: np.ndarray, colors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filtered = cv2.bilateralFilter(rgb, 7, 36, 36)
    lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
    pixels = lab[mask].reshape((-1, 3)).astype(np.float32)
    if len(pixels) == 0:
        raise ValueError("Maschera vuota: impossibile quantizzare i colori.")
    k = max(1, min(colors, len(np.unique(pixels, axis=0))))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.8)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    centers_lab = centers.astype(np.uint8).reshape((1, k, 3))
    centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB).reshape((k, 3)).astype(np.float32)
    full = np.full(mask.shape, -1, dtype=np.int32)
    full[mask] = labels.ravel().astype(np.int32)
    return full, centers_rgb, filtered


def contour_to_path(contour: np.ndarray, simplify: float, curve_strength: float) -> str | None:
    area = abs(cv2.contourArea(contour))
    if area < 1:
        return None
    epsilon = max(0.15, simplify)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape((-1, 2)).astype(np.float32)
    if len(approx) < 3:
        return None

    if len(approx) < 6 or curve_strength <= 0:
        points = " L ".join(f"{fmt(x)} {fmt(y)}" for x, y in approx)
        return f"M {points} Z"

    pts = approx.tolist()
    commands = [f"M {fmt(pts[0][0])} {fmt(pts[0][1])}"]
    strength = max(0.0, min(curve_strength, 0.5))
    n = len(pts)
    for i in range(n):
        p0 = np.array(pts[(i - 1) % n], dtype=np.float32)
        p1 = np.array(pts[i], dtype=np.float32)
        p2 = np.array(pts[(i + 1) % n], dtype=np.float32)
        p3 = np.array(pts[(i + 2) % n], dtype=np.float32)
        c1 = p1 + (p2 - p0) * strength / 3.0
        c2 = p2 - (p3 - p1) * strength / 3.0
        commands.append(
            "C "
            f"{fmt(c1[0])} {fmt(c1[1])} "
            f"{fmt(c2[0])} {fmt(c2[1])} "
            f"{fmt(p2[0])} {fmt(p2[1])}"
        )
    commands.append("Z")
    return " ".join(commands)


def mask_to_curve_paths(mask: np.ndarray, min_area: int, simplify: float, curve_strength: float) -> list[dict]:
    bitmap = (mask.astype(np.uint8) * 255)
    contours, hierarchy = cv2.findContours(bitmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    relations = hierarchy[0]

    children: dict[int, list[int]] = {}
    roots: list[int] = []
    for index, relation in enumerate(relations):
        parent = int(relation[3])
        if parent == -1:
            roots.append(index)
        else:
            children.setdefault(parent, []).append(index)

    def descendants(index: int) -> list[int]:
        result: list[int] = []
        stack = list(children.get(index, []))
        while stack:
            child = stack.pop()
            result.append(child)
            stack.extend(children.get(child, []))
        return result

    paths: list[dict] = []
    for index in sorted(roots, key=lambda item: cv2.contourArea(contours[item]), reverse=True):
        contour = contours[index]
        area = float(abs(cv2.contourArea(contour)))
        if area < min_area:
            continue
        parts: list[str] = []
        point_count = 0
        for contour_index in [index] + descendants(index):
            part = contour_to_path(contours[contour_index], simplify, curve_strength)
            if part:
                parts.append(part)
                point_count += int(len(contours[contour_index]))
        if parts:
            paths.append({"d": " ".join(parts), "area": area, "points": point_count})
    return paths


def write_svg(path: Path, width: int, height: int, layers: list[dict], background: str) -> None:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <rect width="100%" height="100%" fill="{background}"/>',
    ]
    for layer in layers:
        lines.append(
            f'  <g id="{layer["id"]}" data-role="{layer["role"]}" data-area="{layer["area"]}" '
            f'fill="{layer["color"]}" stroke="none">'
        )
        for path_item in layer["paths"]:
            lines.append(f'    <path d="{path_item["d"]}" fill-rule="evenodd"/>')
        lines.append("  </g>")
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    out_dir = Path(args.output) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_rgb(image_path, args.max_size)
    rgb = np.array(image)
    work_mask, mask_mode = choose_work_mask(rgb, args.bg_tolerance, args.mode)
    work_mask = smooth_mask(work_mask, args.subject_close, 0)

    labels, centers_rgb, _ = quantize_lab(rgb, work_mask, args.colors)
    counts = np.bincount(labels[work_mask], minlength=len(centers_rgb))
    order = [int(i) for i in np.argsort(-counts)]
    base_label = order[0]
    background = "#ffffff" if mask_mode.startswith("subject") else hex_color(centers_rgb[base_label])

    layers: list[dict] = []
    if args.base_mode == "subject":
        base_paths = mask_to_curve_paths(work_mask, args.min_base_area, args.base_simplify, args.curve_strength)
        layers.append(
            {
                "id": "base-subject" if mask_mode.startswith("subject") else f"base-color-{base_label}",
                "role": "base",
                "label": base_label,
                "color": hex_color(centers_rgb[base_label]),
                "area": int(work_mask.sum()),
                "paths": base_paths,
            }
        )
    else:
        for label in order:
            if label == base_label and not mask_mode.startswith("subject"):
                continue
            region = (labels == label) & work_mask
            region = remove_small_components(region, args.min_region_area)
            region = smooth_mask(region, args.close, args.open)
            area = int(region.sum())
            if area < args.min_region_area:
                continue
            paths = mask_to_curve_paths(region, args.min_region_area, args.base_simplify, args.curve_strength)
            if not paths:
                continue
            layers.append(
                {
                    "id": f"base-color-{label}",
                    "role": "base",
                    "label": label,
                    "color": hex_color(centers_rgb[label]),
                    "area": area,
                    "paths": paths,
                }
            )

    if args.base_mode == "subject":
        for label in order:
            if label == base_label and mask_mode.startswith("subject"):
                continue
            region = (labels == label) & work_mask
            if label == base_label and not mask_mode.startswith("subject"):
                continue
            region = remove_small_components(region, args.min_region_area)
            region = smooth_mask(region, args.close, args.open)
            area = int(region.sum())
            if area < args.min_region_area:
                continue
            paths = mask_to_curve_paths(region, args.min_region_area, args.simplify, args.curve_strength)
            if not paths:
                continue
            layers.append(
                {
                    "id": f"color-{label}",
                    "role": "overlay",
                    "label": label,
                    "color": hex_color(centers_rgb[label]),
                    "area": area,
                    "paths": paths,
                }
            )

    write_svg(out_dir / "composition_color_trace.svg", image.width, image.height, layers, background)

    preview = np.full(rgb.shape, 255, dtype=np.uint8)
    if not mask_mode.startswith("subject"):
        preview[:] = np.clip(centers_rgb[base_label], 0, 255).astype(np.uint8)
    for layer in layers:
        label = int(layer["label"])
        preview[(labels == label) & work_mask] = np.clip(centers_rgb[label], 0, 255).astype(np.uint8)
    Image.fromarray(preview, mode="RGB").save(out_dir / "color_trace_preview.png")

    report = {
        "input": str(image_path),
        "output": str(out_dir / "composition_color_trace.svg"),
        "mask_mode": mask_mode,
        "size": {"width": image.width, "height": image.height},
        "parameters": {
            "colors": args.colors,
            "min_region_area": args.min_region_area,
            "simplify": args.simplify,
            "base_simplify": args.base_simplify,
            "curve_strength": args.curve_strength,
            "close": args.close,
            "open": args.open,
            "base_mode": args.base_mode,
        },
        "layers": [
            {
                "id": layer["id"],
                "role": layer["role"],
                "color": layer["color"],
                "area": layer["area"],
                "path_count": len(layer["paths"]),
            }
            for layer in layers
        ],
    }
    (out_dir / "color_trace_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Color trace salvato: {out_dir / 'composition_color_trace.svg'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vector tracing sperimentale basato su colori e contorni curvi.")
    parser.add_argument("image")
    parser.add_argument("--output", default="output")
    parser.add_argument("--colors", type=int, default=10)
    parser.add_argument("--base-mode", choices=["partition", "subject"], default="partition")
    parser.add_argument("--mode", choices=["auto", "subject", "full"], default="auto")
    parser.add_argument("--bg-tolerance", type=float, default=35.0)
    parser.add_argument("--min-region-area", type=int, default=45)
    parser.add_argument("--min-base-area", type=int, default=80)
    parser.add_argument("--max-size", type=int, default=1100)
    parser.add_argument("--simplify", type=float, default=1.0)
    parser.add_argument("--base-simplify", type=float, default=1.4)
    parser.add_argument("--curve-strength", type=float, default=0.32)
    parser.add_argument("--close", type=int, default=1)
    parser.add_argument("--open", type=int, default=0)
    parser.add_argument("--subject-close", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
