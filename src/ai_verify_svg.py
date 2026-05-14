from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path


PROMPT = """Confronta l'immagine originale, il piano dei livelli e lo SVG generato per ricamo.

Devi fare un controllo di veridicita' e usabilita':
- il soggetto principale e' corretto?
- lo sfondo/ombra e' stato gestito correttamente?
- le basi coprono tutto cio' che devono coprire?
- i dettagli importanti sono stati mantenuti o persi?
- ci sono forme inventate non presenti nell'immagine?
- i bordi sembrano troppo seghettati, troppo semplificati o poco aderenti?
- la stratificazione e' sensata per ricamo?
- i dettagli/sfumature si alternano in modo utile o si coprono male?

Non essere diplomatico: segnala problemi concreti e proponi manipolatori/azioni correttive.
Valuta lo SVG come tracciato tecnico per ricamo, non come immagine artistica.

Nota: ricevi lo SVG come testo. Usa il contenuto del file per capire layer, path, colori e ordine.
"""


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "score",
        "subject_reading",
        "background_shadow",
        "layering",
        "geometry",
        "detail_retention",
        "embroidery_readiness",
        "required_actions",
        "suggested_parameters",
    ],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["reject", "weak", "promising", "good"],
        },
        "score": {"type": "integer"},
        "subject_reading": {"type": "string"},
        "background_shadow": {"type": "string"},
        "layering": {"type": "string"},
        "geometry": {"type": "string"},
        "detail_retention": {"type": "string"},
        "embroidery_readiness": {"type": "string"},
        "required_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["priority", "action", "reason"],
                "properties": {
                    "priority": {"type": "integer"},
                    "action": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "suggested_parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "edge_smooth",
                "close_pixels",
                "min_region_area",
                "overlap_mode",
                "shadow_mode",
                "notes",
            ],
            "properties": {
                "edge_smooth": {"type": "number"},
                "close_pixels": {"type": "integer"},
                "min_region_area": {"type": "integer"},
                "overlap_mode": {"type": "string"},
                "shadow_mode": {"type": "string"},
                "notes": {"type": "string"},
            },
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


def trimmed_text(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n<!-- TRUNCATED_FOR_VERIFICATION -->"


def call_openai(image_path: Path, svg_path: Path, plan_path: Path | None, model: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Manca OPENAI_API_KEY nell'ambiente.")

    plan_text = trimmed_text(plan_path, 18000) if plan_path and plan_path.exists() else "{}"
    svg_text = trimmed_text(svg_path, 35000)
    user_text = (
        f"{PROMPT}\n\n"
        f"PIANO_LIVELLI_JSON:\n{plan_text}\n\n"
        f"SVG_GENERATO:\n{svg_text}"
    )

    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": image_to_data_url(image_path)},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "embroidery_svg_verification",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica AI di fedelta' e usabilita' ricamo di uno SVG.")
    parser.add_argument("image", help="Immagine originale.")
    parser.add_argument("svg", help="SVG generato da verificare.")
    parser.add_argument("--plan", help="JSON piano livelli usato per generare lo SVG.")
    parser.add_argument("--output", help="File report JSON. Default: accanto allo SVG.")
    parser.add_argument("--model", default="gpt-5.2")
    args = parser.parse_args()

    load_env_file()
    image_path = Path(args.image)
    svg_path = Path(args.svg)
    plan_path = Path(args.plan) if args.plan else None
    report = call_openai(image_path, svg_path, plan_path, args.model)
    output_path = Path(args.output) if args.output else svg_path.with_name(svg_path.stem + "_verification.json")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verifica AI salvata: {output_path}")


if __name__ == "__main__":
    main()

