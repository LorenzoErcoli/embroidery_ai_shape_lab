@echo off
if "%~1"=="" (
  echo Uso: run_clean_trace.bat input\immagine.png
  echo Richiede output\nome\composition_ai.svg e output\nome\ai_plan.json gia' generati.
  exit /b 1
)
set PY=C:\Users\l.ercoli\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
for %%F in ("%~1") do set STEM=%%~nF
"%PY%" src\clean_trace_svg.py "output\%STEM%\composition_ai.svg" --output "output\%STEM%\composition_ai_plan_clean.svg" --plan "output\%STEM%\ai_plan.json" --simplify 0.45 --min-base-area 8 --min-overlay-area 3 --min-detail-area 1
