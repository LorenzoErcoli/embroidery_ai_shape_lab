from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


PYTHON_EXE = r"C:\Users\l.ercoli\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"


PRESETS = [
    {
        "name": "balanced",
        "args": [
            "--color-tolerance",
            "52",
            "--min-region-area",
            "160",
            "--simplify",
            "6",
            "--edge-smooth",
            "1.4",
            "--close-pixels",
            "2",
            "--overlap-mode",
            "details-only",
            "--shadow-mode",
            "force",
            "--shape-prior",
            "auto",
        ],
    },
    {
        "name": "macro",
        "args": [
            "--color-tolerance",
            "56",
            "--min-region-area",
            "900",
            "--simplify",
            "8",
            "--edge-smooth",
            "1.8",
            "--close-pixels",
            "6",
            "--overlap-mode",
            "details-only",
            "--shadow-mode",
            "force",
            "--shape-prior",
            "auto",
            "--max-overlay-components",
            "6",
            "--max-outline-components",
            "8",
            "--max-detail-components",
            "4",
        ],
    },
    {
        "name": "clean",
        "args": [
            "--color-tolerance",
            "46",
            "--min-region-area",
            "350",
            "--simplify",
            "7",
            "--edge-smooth",
            "1.1",
            "--close-pixels",
            "4",
            "--overlap-mode",
            "none",
            "--shadow-mode",
            "force",
            "--shape-prior",
            "auto",
            "--max-overlay-components",
            "8",
            "--max-outline-components",
            "6",
            "--max-detail-components",
            "5",
        ],
    },
]


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def score_report(path: Path) -> tuple[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("score", 0)), str(data.get("verdict", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Esegue conversione e verifica AI su piu' preset.")
    parser.add_argument("image")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--target-score", type=int, default=60)
    parser.add_argument("--model", default="gpt-5.2")
    args = parser.parse_args()

    root = Path.cwd()
    image = Path(args.image)
    stem = image.stem
    base_out = root / "output" / stem
    base_out.mkdir(parents=True, exist_ok=True)
    plan_path = base_out / "ai_plan.json"

    if not plan_path.exists():
        run([PYTHON_EXE, "src/ai_embroidery_plan.py", str(image), "--model", args.model], root)

    attempts: list[dict] = []
    for index, preset in enumerate(PRESETS[: args.max_attempts], start=1):
        attempt_dir = base_out / f"attempt_{index}_{preset['name']}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                PYTHON_EXE,
                "src/ai_plan_to_svg.py",
                str(image),
                str(plan_path),
                "--output",
                str(base_out / f"attempt_{index}_{preset['name']}_out"),
                *preset["args"],
            ],
            root,
        )
        generated_dir = base_out / f"attempt_{index}_{preset['name']}_out" / stem
        svg_path = generated_dir / "composition_ai.svg"
        report_path = generated_dir / "ai_svg_report.json"
        shutil.copy2(svg_path, attempt_dir / "composition_ai.svg")
        shutil.copy2(report_path, attempt_dir / "ai_svg_report.json")

        verification_path = attempt_dir / "composition_ai_verification.json"
        run(
            [
                PYTHON_EXE,
                "src/ai_verify_svg.py",
                str(image),
                str(attempt_dir / "composition_ai.svg"),
                "--plan",
                str(plan_path),
                "--output",
                str(verification_path),
                "--model",
                args.model,
            ],
            root,
        )
        score, verdict = score_report(verification_path)
        attempts.append(
            {
                "attempt": index,
                "preset": preset["name"],
                "score": score,
                "verdict": verdict,
                "dir": str(attempt_dir),
            }
        )
        if score >= args.target_score:
            break

    best = max(attempts, key=lambda item: item["score"])
    best_dir = Path(best["dir"])
    shutil.copy2(best_dir / "composition_ai.svg", base_out / "composition_ai.svg")
    shutil.copy2(best_dir / "ai_svg_report.json", base_out / "ai_svg_report.json")
    shutil.copy2(best_dir / "composition_ai_verification.json", base_out / "composition_ai_verification.json")
    summary = {"best": best, "attempts": attempts}
    (base_out / "iteration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

