from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path


PROMPT = """Analizza questa immagine per preparare una scomposizione adatta al ricamo.

Obiettivo: non fare un ricalco frammentato. Devi proporre una gerarchia di livelli:
1. sfondo esterno da ignorare;
2. soggetto principale con sagoma unica e colore base;
3. decorazioni o riempimenti sopra il soggetto;
4. dettagli piccoli da tenere solo se ricamabili.

Ragiona come un tecnico grafico per ricamo: preferisci poche campiture chiuse, colori accorpati,
base sotto e dettagli sopra. Evita texture fotografiche, ombre morbide e micro-frammenti.
I colori devono essere stimati in HEX e pensati come fili/campiture.
"""


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "subject_type",
        "background_handling",
        "composition_strategy",
        "layers",
        "discard_rules",
        "risks",
    ],
    "properties": {
        "subject_type": {"type": "string"},
        "background_handling": {"type": "string"},
        "composition_strategy": {"type": "string"},
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
                    "embroidery_use",
                    "shape_description",
                    "include",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["base", "overlay", "outline", "detail", "discard"],
                    },
                    "color_hex": {"type": "string"},
                    "priority": {"type": "integer"},
                    "embroidery_use": {"type": "string"},
                    "shape_description": {"type": "string"},
                    "include": {"type": "boolean"},
                },
            },
        },
        "discard_rules": {
            "type": "array",
            "items": {"type": "string"},
        },
        "risks": {
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
                "name": "embroidery_shape_plan",
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
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Errore OpenAI {exc.code}: {detail}") from exc

    text = extract_output_text(payload)
    if not text:
        raise SystemExit("La risposta OpenAI non contiene output_text.")
    plan = json.loads(text)
    plan["_openai"] = {"model": model, "response_id": payload.get("id")}
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un piano AI di livelli ricamo da immagine.")
    parser.add_argument("image", help="Percorso immagine input.")
    parser.add_argument("--output", default="output", help="Cartella output.")
    parser.add_argument("--model", default="gpt-5.2", help="Modello vision OpenAI.")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.output) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = call_openai(image_path, args.model)
    target = out_dir / "ai_plan.json"
    target.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Piano AI salvato: {target}")


if __name__ == "__main__":
    main()

