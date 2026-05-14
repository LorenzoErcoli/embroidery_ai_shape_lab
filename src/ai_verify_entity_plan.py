from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path


PROMPT = """Valuta SOLO la lettura AI e il piano di entita' per ricamo.

Non valutare lo SVG e non giudicare la qualita' dei bordi vettoriali.
Devi dire se il piano AI capisce bene l'immagine:
- soggetti principali;
- parti interne;
- sfondo/oggetti da escludere;
- livelli base/overlay/detail;
- dettagli importanti mantenuti o scartati;
- relazioni parent/child e ordine logico.

Se il piano e' buono ma la futura segmentazione potrebbe essere difficile, dichiaralo separatamente.
"""


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "score",
        "reading_quality",
        "entity_coverage",
        "layering_quality",
        "problems",
        "segmentation_risks",
        "required_plan_fixes",
    ],
    "properties": {
        "verdict": {"type": "string", "enum": ["good", "weak", "reject"]},
        "score": {"type": "integer"},
        "reading_quality": {"type": "string"},
        "entity_coverage": {"type": "string"},
        "layering_quality": {"type": "string"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "segmentation_risks": {"type": "array", "items": {"type": "string"}},
        "required_plan_fixes": {"type": "array", "items": {"type": "string"}},
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


def call_openai(image_path: Path, plan: dict, model: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Manca OPENAI_API_KEY nell'ambiente.")
    text = PROMPT + "\n\nPiano entita' JSON:\n" + json.dumps(plan, ensure_ascii=False, indent=2)
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_image", "image_url": image_to_data_url(image_path)},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "entity_plan_verification",
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
    output_text = extract_output_text(payload)
    if not output_text:
        raise SystemExit("La risposta OpenAI non contiene output_text.")
    result = json.loads(output_text)
    result["_openai"] = {"model": model, "response_id": payload.get("id")}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica solo la qualita' del piano entita' AI.")
    parser.add_argument("image")
    parser.add_argument("plan")
    parser.add_argument("--output")
    parser.add_argument("--model", default="gpt-5.2")
    args = parser.parse_args()

    load_env_file()
    image_path = Path(args.image)
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = call_openai(image_path, plan, args.model)
    output = Path(args.output) if args.output else plan_path.with_name("entity_plan_verification.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Verifica piano AI salvata: {output}")


if __name__ == "__main__":
    main()
