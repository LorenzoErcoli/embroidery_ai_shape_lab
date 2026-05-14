from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


COMMAND_RE = re.compile(r"[MLCZ]|-?\d+(?:\.\d+)?")
ET.register_namespace("", "http://www.w3.org/2000/svg")


def parse_poly_path(d: str) -> list[tuple[float, float]] | None:
    tokens = COMMAND_RE.findall(d)
    if not tokens or any(token in {"C", "Q"} for token in tokens):
        return None
    points: list[tuple[float, float]] = []
    index = 0
    command = ""
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token in {"M", "L", "Z"}:
            command = token
            if command == "Z":
                continue
        if command not in {"M", "L"}:
            return None
        if token in {"M", "L"}:
            if index + 1 > len(tokens):
                return None
            x = float(tokens[index])
            y = float(tokens[index + 1])
            index += 2
        else:
            if index >= len(tokens):
                return None
            x = float(token)
            y = float(tokens[index])
            index += 1
        points.append((x, y))
    if len(points) < 3:
        return None
    if points[0] == points[-1]:
        points = points[:-1]
    return points


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def bbox_area(points: list[tuple[float, float]]) -> float:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def point_line_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if dx == 0 and dy == 0:
        return math.dist(point, start)
    return abs(dy * px - dx * py + ex * sy - ey * sx) / math.hypot(dx, dy)


def rdp_open(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    max_dist = -1.0
    max_index = 0
    for index in range(1, len(points) - 1):
        dist = point_line_distance(points[index], start, end)
        if dist > max_dist:
            max_dist = dist
            max_index = index
    if max_dist > epsilon:
        left = rdp_open(points[: max_index + 1], epsilon)
        right = rdp_open(points[max_index:], epsilon)
        return left[:-1] + right
    return [start, end]


def simplify_closed(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) < 8 or epsilon <= 0:
        return points
    closed = points + [points[0]]
    simplified = rdp_open(closed, epsilon)
    if simplified and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]
    return simplified if len(simplified) >= 3 else points


def remove_collinear(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 4:
        return points
    result: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        prev = points[index - 1]
        nxt = points[(index + 1) % len(points)]
        if point_line_distance(point, prev, nxt) < 0.1:
            continue
        result.append(point)
    return result if len(result) >= 3 else points


def path_from_points(points: list[tuple[float, float]]) -> str:
    def fmt(value: float) -> str:
        if abs(value - round(value)) < 0.05:
            return str(int(round(value)))
        return f"{value:.1f}"

    return "M " + " L ".join(f"{fmt(x)} {fmt(y)}" for x, y in points) + " Z"


def min_area_for_role(role: str, base: float, overlay: float, detail: float) -> float:
    if role == "base":
        return base
    if role == "detail":
        return detail
    return overlay


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def infer_cap(layer: dict) -> int | None:
    role = layer.get("role", "overlay")
    text = f'{layer.get("name", "")} {layer.get("shape_description", "")} {layer.get("embroidery_use", "")}'.lower()
    if role == "base":
        return 1
    if "12" in text and "20" in text:
        return 20
    if "30" in text and "60" in text:
        return 60
    ranges = re.findall(r"(\d+)\s*[-–]\s*(\d+)", text)
    if ranges:
        return max(int(end) for _, end in ranges)
    if role == "outline":
        return 20
    if role == "detail":
        return 24
    return 6


def load_caps(plan_path: Path | None) -> dict[str, int]:
    if not plan_path:
        return {}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    caps: dict[str, int] = {}
    for layer in plan.get("layers", []):
        if not layer.get("include"):
            continue
        cap = infer_cap(layer)
        if cap is not None:
            caps[normalize_name(layer.get("name", ""))] = cap
    return caps


def cap_for_group(group: ET.Element, caps: dict[str, int]) -> int | None:
    name = normalize_name(group.attrib.get("data-ai-name", "") or group.attrib.get("id", ""))
    if name in caps:
        return caps[name]
    for key, cap in caps.items():
        if key and (key in name or name in key):
            return cap
    return None


def clean_svg(
    input_path: Path,
    output_path: Path,
    simplify: float,
    min_base: float,
    min_overlay: float,
    min_detail: float,
    caps: dict[str, int],
    use_caps: bool,
) -> dict:
    tree = ET.parse(input_path)
    root = tree.getroot()
    removed = 0
    simplified = 0
    kept = 0
    capped = 0
    for group in root.findall(".//{http://www.w3.org/2000/svg}g"):
        role = group.attrib.get("data-role", "overlay")
        min_area = min_area_for_role(role, min_base, min_overlay, min_detail)
        candidates: list[tuple[ET.Element, list[tuple[float, float]] | None, float]] = []
        for path in list(group.findall("{http://www.w3.org/2000/svg}path")):
            d = path.attrib.get("d", "")
            points = parse_poly_path(d)
            if points is None:
                candidates.append((path, None, float("inf")))
                continue
            area = polygon_area(points)
            if area < min_area or bbox_area(points) < min_area:
                group.remove(path)
                removed += 1
                continue
            candidates.append((path, points, area))

        cap = cap_for_group(group, caps) if use_caps else None
        if cap is not None and len(candidates) > cap:
            keep = set(id(item[0]) for item in sorted(candidates, key=lambda item: item[2], reverse=True)[:cap])
            for path, _, _ in candidates:
                if id(path) not in keep:
                    group.remove(path)
                    removed += 1
                    capped += 1
            candidates = [item for item in candidates if id(item[0]) in keep]

        for path, points, _ in candidates:
            if points is None:
                kept += 1
                continue
            cleaned = remove_collinear(simplify_closed(points, simplify))
            if len(cleaned) != len(points):
                simplified += 1
                path.set("d", path_from_points(cleaned))
            kept += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return {"removed_paths": removed, "capped_paths": capped, "simplified_paths": simplified, "kept_paths": kept}


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process conservativo per SVG trace-like.")
    parser.add_argument("input_svg")
    parser.add_argument("--output")
    parser.add_argument("--simplify", type=float, default=0.45)
    parser.add_argument("--min-base-area", type=float, default=8)
    parser.add_argument("--min-overlay-area", type=float, default=3)
    parser.add_argument("--min-detail-area", type=float, default=1)
    parser.add_argument("--plan", help="Piano AI JSON per cap path per layer.")
    parser.add_argument("--use-plan-caps", action="store_true", help="Limita il numero di path usando il piano AI. Disattivo di default per preservare dettagli e colori.")
    args = parser.parse_args()

    input_path = Path(args.input_svg)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_clean.svg")
    caps = load_caps(Path(args.plan)) if args.plan else {}
    report = clean_svg(input_path, output_path, args.simplify, args.min_base_area, args.min_overlay_area, args.min_detail_area, caps, args.use_plan_caps)
    report_path = output_path.with_suffix(".json")
    report_path.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
    print(f"SVG pulito salvato: {output_path}")


if __name__ == "__main__":
    main()
