from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Region:
    label: int
    color: tuple[int, int, int]
    area: int
    path_count: int
    is_base: bool = False


def load_rgb(path: Path, max_size: int) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    image = background.convert("RGB")
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def detect_subject_mask(rgb: np.ndarray, tolerance: float) -> np.ndarray:
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
    bg_color = np.median(corners, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg_color, axis=2)
    bg_candidate = dist <= tolerance

    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if bg_candidate[0, x]:
            q.append((0, x))
        if bg_candidate[h - 1, x]:
            q.append((h - 1, x))
    for y in range(h):
        if bg_candidate[y, 0]:
            q.append((y, 0))
        if bg_candidate[y, w - 1]:
            q.append((y, w - 1))

    while q:
        y, x = q.popleft()
        if exterior[y, x] or not bg_candidate[y, x]:
            continue
        exterior[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not exterior[ny, nx]:
                q.append((ny, nx))

    subject = ~exterior
    return keep_largest_component(subject)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
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
            if len(cells) > len(best):
                best = cells
    result = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        result[y, x] = True
    return result


def kmeans_colors(pixels: np.ndarray, k: int, iterations: int = 18) -> tuple[np.ndarray, np.ndarray]:
    if len(pixels) == 0:
        raise ValueError("La maschera soggetto e' vuota.")

    unique = np.unique(pixels, axis=0)
    k = min(k, len(unique))
    if k == 1:
        return np.zeros(len(pixels), dtype=np.int32), unique.astype(np.float32)

    luminance = unique @ np.array([0.2126, 0.7152, 0.0722])
    order = np.argsort(luminance)
    centers = unique[order[np.linspace(0, len(unique) - 1, k).astype(int)]].astype(np.float32)

    px = pixels.astype(np.float32)
    labels = np.zeros(len(px), dtype=np.int32)
    for _ in range(iterations):
        distances = np.sum((px[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        for idx in range(k):
            members = px[labels == idx]
            if len(members):
                centers[idx] = members.mean(axis=0)
    return labels, centers


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = np.zeros_like(mask, dtype=bool)
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
            if len(cells) >= min_area:
                for y, x in cells:
                    result[y, x] = True
    return result


def mask_to_paths(mask: np.ndarray, simplify: int) -> list[str]:
    h, w = mask.shape
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add_edge(start: tuple[int, int], end: tuple[int, int]) -> None:
        edges.setdefault(start, []).append(end)

    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            if y == 0 or not mask[y - 1, x]:
                add_edge((x, y), (x + 1, y))
            if x == w - 1 or not mask[y, x + 1]:
                add_edge((x + 1, y), (x + 1, y + 1))
            if y == h - 1 or not mask[y + 1, x]:
                add_edge((x + 1, y + 1), (x, y + 1))
            if x == 0 or not mask[y, x - 1]:
                add_edge((x, y + 1), (x, y))

    paths: list[str] = []
    while edges:
        start = next(iter(edges))
        points = [start]
        current = start
        guard = 0
        while guard < h * w * 8:
            guard += 1
            next_points = edges.get(current)
            if not next_points:
                break
            nxt = next_points.pop()
            if not next_points:
                del edges[current]
            points.append(nxt)
            current = nxt
            if current == start:
                break
        clean = simplify_points(points, simplify)
        if len(clean) >= 4:
            command = "M " + " L ".join(f"{x} {y}" for x, y in clean[:-1]) + " Z"
            paths.append(command)
    return paths


def simplify_points(points: list[tuple[int, int]], step: int) -> list[tuple[int, int]]:
    if len(points) <= 4:
        return points
    reduced = [points[0]]
    for i in range(1, len(points) - 1):
        prev = reduced[-1]
        curr = points[i]
        nxt = points[i + 1]
        same_line = (prev[0] == curr[0] == nxt[0]) or (prev[1] == curr[1] == nxt[1])
        far_enough = math.dist(prev, curr) >= step
        if not same_line and far_enough:
            reduced.append(curr)
    if reduced[-1] != points[-1]:
        reduced.append(points[-1])
    return reduced


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_preview(labels_full: np.ndarray, centers: np.ndarray, subject: np.ndarray, path: Path) -> None:
    h, w = labels_full.shape
    preview = np.full((h, w, 3), 245, dtype=np.uint8)
    colors = np.clip(centers, 0, 255).astype(np.uint8)
    preview[subject] = colors[labels_full[subject]]
    Image.fromarray(preview, mode="RGB").save(path)


def write_svg(path: Path, width: int, height: int, regions: list[Region], path_map: dict[int, list[str]]) -> None:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '  <rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for region in regions:
        color = "#{:02x}{:02x}{:02x}".format(*region.color)
        role = "base" if region.is_base else "overlay"
        lines.append(f'  <g id="{role}-{region.label}" data-area="{region.area}" fill="{color}" stroke="#111111" stroke-width="0.6">')
        for d in path_map.get(region.label, []):
            lines.append(f'    <path d="{d}"/>')
        lines.append("  </g>")
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> None:
    input_path = Path(args.image)
    out_dir = Path(args.output) / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_rgb(input_path, args.max_size)
    rgb = np.array(image)
    subject = detect_subject_mask(rgb, args.bg_tolerance)
    subject_pixels = rgb[subject]

    labels, centers = kmeans_colors(subject_pixels, args.colors)
    labels_full = np.full(subject.shape, -1, dtype=np.int32)
    labels_full[subject] = labels

    counts = np.bincount(labels, minlength=len(centers))
    base_label = int(np.argmax(counts))
    centers_uint = np.clip(np.rint(centers), 0, 255).astype(np.uint8)

    path_map: dict[int, list[str]] = {}
    regions: list[Region] = []

    base_paths = mask_to_paths(subject, args.simplify)
    path_map[base_label] = base_paths
    regions.append(
        Region(
            label=base_label,
            color=tuple(int(v) for v in centers_uint[base_label]),
            area=int(subject.sum()),
            path_count=len(base_paths),
            is_base=True,
        )
    )

    for label in np.argsort(-counts):
        label = int(label)
        if label == base_label:
            continue
        region_mask = remove_small_components(labels_full == label, args.min_region_area)
        area = int(region_mask.sum())
        if area == 0:
            continue
        paths = mask_to_paths(region_mask, args.simplify)
        if not paths:
            continue
        path_map[label] = paths
        regions.append(
            Region(
                label=label,
                color=tuple(int(v) for v in centers_uint[label]),
                area=area,
                path_count=len(paths),
            )
        )

    save_mask(subject, out_dir / "subject_mask.png")
    save_preview(labels_full, centers, subject, out_dir / "regions_preview.png")
    write_svg(out_dir / "composition.svg", image.width, image.height, regions, path_map)

    report = {
        "input": str(input_path),
        "size": {"width": image.width, "height": image.height},
        "parameters": {
            "colors": args.colors,
            "bg_tolerance": args.bg_tolerance,
            "min_region_area": args.min_region_area,
            "max_size": args.max_size,
            "simplify": args.simplify,
        },
        "subject_area": int(subject.sum()),
        "regions": [
            {
                "label": r.label,
                "role": "base" if r.is_base else "overlay",
                "color_rgb": list(r.color),
                "color_hex": "#{:02x}{:02x}{:02x}".format(*r.color),
                "area": r.area,
                "path_count": r.path_count,
            }
            for r in regions
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Analisi completata: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmenta soggetto e regioni colore per test ricamo.")
    parser.add_argument("image", help="Percorso immagine input.")
    parser.add_argument("--output", default="output", help="Cartella output.")
    parser.add_argument("--colors", type=int, default=6, help="Numero cluster colore nel soggetto.")
    parser.add_argument("--bg-tolerance", type=float, default=35.0, help="Tolleranza colore sfondo dai bordi.")
    parser.add_argument("--min-region-area", type=int, default=80, help="Area minima componenti overlay.")
    parser.add_argument("--max-size", type=int, default=900, help="Lato massimo immagine in pixel.")
    parser.add_argument("--simplify", type=int, default=2, help="Distanza minima tra vertici mantenuti.")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
