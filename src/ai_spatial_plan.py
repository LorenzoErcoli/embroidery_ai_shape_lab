from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path


PROMPT = """Analizza l'immagine per una vettorializzazione da ricamo guidata da maschere.

Non devi disegnare tu i tracciati finali. Devi fornire un piano spaziale:
- soggetto principale e bounding box;
- sfondo/ombra da escludere;
- layer da ricamare;
- per ogni layer: colore, ruolo, area dell'immagine dove cercarlo, numero massimo di componenti, cosa scartare.

Coordinate normalizzate 0-1000:
- x1/y1 angolo alto-sinistra
- x2/y2 angolo basso-destra

Regole:
- Le bbox devono essere strette abbastanza da evitare falsi positivi in altre zone.
- Se due elementi hanno lo stesso colore ma sono semanticamente diversi, separali in layer diversi con bbox diverse.
- Per ricamo preferisci poche componenti grandi.
- Testi piccoli, texture, puntini, riflessi e ombre morbide vanno scartati salvo richiesta esplicita.
- Per soggetti come palloni, libri, frutta, bici, fiori: distingui base, parti principali, dettagli ricamabili.
- Indica quando una forma dovrebbe essere ricostruita geometricamente:
  ellipse, rings, wheel_rings, bars, tube_bars, seed_pattern, smooth_blob, outline_band, book_block, star_panels, freeform.
"""


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["subject", "background", "layers", "global_rules"],
    "properties": {
        "subject": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "bbox", "shape_hint", "notes"],
            "properties": {
                "type": {"type": "string"},
                "bbox": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x1", "y1", "x2", "y2"],
                    "properties": {
                        "x1": {"type": "integer"},
                        "y1": {"type": "integer"},
                        "x2": {"type": "integer"},
                        "y2": {"type": "integer"},
                    },
                },
                "shape_hint": {
                    "type": "string",
                    "enum": ["ellipse", "smooth_blob", "multi_object", "thin_structure", "stacked_blocks"],
                },
                "notes": {"type": "string"},
            },
        },
        "background": {
            "type": "object",
            "additionalProperties": False,
            "required": ["discard", "shadow_notes"],
            "properties": {
                "discard": {"type": "boolean"},
                "shadow_notes": {"type": "string"},
            },
        },
        "layers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "role",
                    "color_hex",
                    "bbox",
                    "shape_hint",
                    "max_components",
                    "min_area_ratio",
                    "priority",
                    "include",
                    "notes",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "enum": ["base", "overlay", "outline", "detail", "discard"]},
                    "color_hex": {"type": "string"},
                    "bbox": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["x1", "y1", "x2", "y2"],
                        "properties": {
                            "x1": {"type": "integer"},
                            "y1": {"type": "integer"},
                            "x2": {"type": "integer"},
                            "y2": {"type": "integer"},
                        },
                    },
                    "shape_hint": {
                        "type": "string",
                        "enum": [
                            "ellipse",
                            "rings",
                            "wheel_rings",
                            "bars",
                            "tube_bars",
                            "seed_pattern",
                            "smooth_blob",
                            "outline_band",
                            "book_block",
                            "star_panels",
                            "freeform"
                        ],
                    },
                    "max_components": {"type": "integer"},
                    "min_area_ratio": {"type": "number"},
                    "priority": {"type": "integer"},
                    "include": {"type": "boolean"},
                    "notes": {"type": "string"},
                },
            },
        },
        "global_rules": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


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
                "name": "spatial_embroidery_plan",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera piano spaziale AI per maschere ricamo.")
    parser.add_argument("image")
    parser.add_argument("--output", default="output")
    parser.add_argument("--model", default="gpt-5.2")
    args = parser.parse_args()

    load_env_file()
    image_path = Path(args.image)
    out_dir = Path(args.output) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = call_openai(image_path, args.model)
    target = out_dir / "spatial_ai_plan.json"
    target.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Piano spaziale AI salvato: {target}")


if __name__ == "__main__":
    main()
