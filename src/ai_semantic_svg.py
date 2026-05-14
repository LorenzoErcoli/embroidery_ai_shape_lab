from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


PROMPT = """Analizza l'immagine e crea una scomposizione vettoriale semantica per ricamo.

Questa volta non devi solo descrivere i livelli: devi produrre poligoni normalizzati 0-1000 per ogni livello incluso.
Obiettivo: SVG ricamabile, non ricalco fotografico.

Regole:
- ignora lo sfondo esterno e le ombre esterne;
- crea una o piu' basi complete che coprano tutto il soggetto utile;
- sopra la base, crea campiture chiuse per grafiche, ombre, luci e dettagli;
- le campiture principali non devono coprirsi tra loro inutilmente: devono alternarsi dove rappresentano dettagli/sfumature;
- preferisci pochi poligoni grandi invece di molti frammenti;
- semplifica texture, testi piccoli, puntinature e micro-dettagli;
- usa coordinate normalizzate: x=0 sinistra, x=1000 destra, y=0 alto, y=1000 basso;
- ogni poligono deve avere punti in ordine, chiuso implicitamente;
- evita meno di 4 punti per poligono;
- usa un numero massimo ragionevole di poligoni per layer.

Il risultato serve come test di maschera AI semantica, quindi meglio una forma pulita e intenzionale che un ricalco seghettato.
"""


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["subject_type", "strategy", "layers", "global_notes"],
    "properties": {
        "subject_type": {"type": "string"},
        "strategy": {"type": "string"},
        "layers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "role",
                    "color_hex",
                    "priority",
                    "stitch_intent",
                    "notes",
                    "polygons",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["base", "overlay", "outline", "detail"],
                    },
                    "color_hex": {"type": "string"},
                    "priority": {"type": "integer"},
                    "stitch_intent": {"type": "string"},
                    "notes": {"type": "string"},
                    "polygons": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["x", "y"],
                                "properties": {
                                    "x": {"type": "integer"},
                                    "y": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
            },
        },
        "global_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def extract_output_text(response: dict) -> str:
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    if not chunks and "output_text" in response:
        return str(response["output_text"])
    return "".join(chunks)


def call_openai(image_path: Path, model: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Manca OPENAI_API_KEY nell'ambiente.")

    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": image_to_data_url(image_path)},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "semantic_embroidery_svg",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Errore OpenAI {exc.code}: {detail}") from exc

    text = extract_output_text(payload)
    if not text:
        raise SystemExit("La risposta OpenAI non contiene output_text.")
    result = json.loads(text)
    result["_openai"] = {"model": model, "response_id": payload.get("id")}
    return result


def parse_hex(value: str) -> tuple[int, int, int]:
    match = HEX_RE.match(value.strip())
    if not match:
        return 180, 180, 180
    raw = match.group(1)
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def point_to_xy(point: dict, width: int, height: int) -> tuple[float, float]:
    x = max(0, min(1000, int(point.get("x", 0)))) / 1000 * width
    y = max(0, min(1000, int(point.get("y", 0)))) / 1000 * height
    return x, y


def polygon_path(points: list[dict], width: int, height: int) -> str | None:
    if len(points) < 4:
        return None
    coords = [point_to_xy(point, width, height) for point in points]
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in coords) + " Z"


def write_svg(plan: dict, image_path: Path, output_path: Path) -> dict:
    with Image.open(image_path) as image:
        width, height = image.size

    layers = sorted(plan.get("layers", []), key=lambda item: item.get("priority", 999))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '  <rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    report_layers: list[dict] = []
    for index, layer in enumerate(layers, start=1):
        color = parse_hex(layer.get("color_hex", ""))
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", layer.get("name", "").strip()).strip("-") or f"layer-{index}"
        paths = []
        for polygon in layer.get("polygons", []):
            path = polygon_path(polygon, width, height)
            if path:
                paths.append(path)
        if not paths:
            continue
        lines.append(
            f'  <g id="semantic-{index}-{name}" data-role="{layer.get("role", "")}" '
            f'data-ai-name="{layer.get("name", "")}" fill="#{color[0]:02x}{color[1]:02x}{color[2]:02x}" '
            'stroke="#111111" stroke-width="0.8" stroke-linejoin="round">'
        )
        lines.append(f'    <title>{layer.get("notes", "")}</title>')
        for path in paths:
            lines.append(f'    <path d="{path}"/>')
        lines.append("  </g>")
        report_layers.append(
            {
                "name": layer.get("name", ""),
                "role": layer.get("role", ""),
                "color_hex": "#{:02x}{:02x}{:02x}".format(*color),
                "polygon_count": len(paths),
                "stitch_intent": layer.get("stitch_intent", ""),
            }
        )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "subject_type": plan.get("subject_type", ""),
        "strategy": plan.get("strategy", ""),
        "layers": report_layers,
        "global_notes": plan.get("global_notes", []),
        "openai": plan.get("_openai", {}),
    }


def load_env_file() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and not os.environ.get(key.strip()):
            os.environ[key.strip()] = value.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera SVG semantico AI con poligoni per layer.")
    parser.add_argument("image", help="Percorso immagine input.")
    parser.add_argument("--output", default="output", help="Cartella output.")
    parser.add_argument("--model", default="gpt-5.2")
    args = parser.parse_args()

    load_env_file()
    image_path = Path(args.image)
    out_dir = Path(args.output) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = call_openai(image_path, args.model)

    semantic_json = out_dir / "semantic_ai_plan.json"
    semantic_svg = out_dir / "semantic_ai.svg"
    semantic_report = out_dir / "semantic_ai_report.json"
    semantic_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    report = write_svg(plan, image_path, semantic_svg)
    semantic_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SVG semantico AI salvato: {semantic_svg}")


if __name__ == "__main__":
    main()

