from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path


PROMPT = """Analizza l'immagine per costruire un piano gerarchico di entita' visive da convertire in maschere SVG per ricamo.

Non devi disegnare primitive e non devi inventare forme. Devi descrivere cosa separare.

Obiettivo:
- riconoscere tutte le entita' visive importanti;
- distinguere oggetti, parti, decorazioni, texture, scritte, riflessi, ombre, sfondo;
- indicare relazioni parent/child: cosa appartiene a cosa e cosa sta sopra cosa;
- produrre livelli ricamabili con poche campiture coerenti.

Coordinate normalizzate 0-1000.

Regole:
- Crea una o piu' entita' root per i soggetti principali. Lo sfondo e gli elementi esclusi devono essere entita' separate con include=false.
- Ogni parte interna deve avere parent_id dell'oggetto a cui appartiene.
- Se un dettaglio e' troppo sottile, testurale, fotografico o non ricamabile, include=false.
- Non usare shape specialistiche tipo bici/libro/pallone. Usa solo ruoli generici: object, part, fill, decoration, detail, texture, text, reflection, shadow, background.
- mask_source deve dire come cercare la maschera:
  subject_region = intera regione dell'entita' dentro bbox;
  color_range = pixels vicini al colore dentro parent/bbox;
  parent_region = eredita/interseca la maschera parent;
  exclude = entita' da scartare.
- Per i detail limita sempre area e componenti: non devono diventare masse enormi.
- Per ricamo preferisci basi grandi, riempimenti sopra e dettagli solo se necessari.
"""


BBOX = {
    "type": "object",
    "additionalProperties": False,
    "required": ["x1", "y1", "x2", "y2"],
    "properties": {
        "x1": {"type": "integer"},
        "y1": {"type": "integer"},
        "x2": {"type": "integer"},
        "y2": {"type": "integer"},
    },
}


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["image_reading", "entities", "layering_rules"],
    "properties": {
        "image_reading": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "parent_id",
                    "name",
                    "entity_type",
                    "role",
                    "include",
                    "priority",
                    "color_hex",
                    "bbox",
                    "mask_source",
                    "max_components",
                    "min_area_ratio",
                    "max_area_ratio",
                    "edge_policy",
                    "notes",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "parent_id": {"type": ["string", "null"]},
                    "name": {"type": "string"},
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "object",
                            "part",
                            "fill",
                            "decoration",
                            "detail",
                            "texture",
                            "text",
                            "reflection",
                            "shadow",
                            "background",
                        ],
                    },
                    "role": {"type": "string", "enum": ["base", "overlay", "detail", "exclude"]},
                    "include": {"type": "boolean"},
                    "priority": {"type": "integer"},
                    "color_hex": {"type": "string"},
                    "bbox": BBOX,
                    "mask_source": {
                        "type": "string",
                        "enum": ["subject_region", "color_range", "parent_region", "exclude"],
                    },
                    "max_components": {"type": "integer"},
                    "min_area_ratio": {"type": "number"},
                    "max_area_ratio": {"type": "number"},
                    "edge_policy": {"type": "string", "enum": ["solid", "smooth", "clean", "fine"]},
                    "notes": {"type": "string"},
                },
            },
        },
        "layering_rules": {"type": "array", "items": {"type": "string"}},
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
                "name": "entity_embroidery_plan",
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
    parser = argparse.ArgumentParser(description="Genera piano AI gerarchico per entita' visive.")
    parser.add_argument("image")
    parser.add_argument("--output", default="output")
    parser.add_argument("--model", default="gpt-5.2")
    args = parser.parse_args()

    load_env_file()
    image_path = Path(args.image)
    out_dir = Path(args.output) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = call_openai(image_path, args.model)
    target = out_dir / "entity_ai_plan.json"
    target.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Piano entita' AI salvato: {target}")


if __name__ == "__main__":
    main()
