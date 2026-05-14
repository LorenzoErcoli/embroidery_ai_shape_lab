@echo off
if "%~1"=="" (
  echo Uso: run_entity_trace.bat input\immagine.png
  echo Richiede un entity_ai_plan.json gia' generato oppure lancia prima run_entity_ai.bat.
  exit /b 1
)
set PY=C:\Users\l.ercoli\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
for %%F in ("%~1") do set STEM=%%~nF
"%PY%" src\entity_plan_to_svg.py "%~1" "output\%STEM%\entity_ai_plan.json" --color-tolerance 72 --edge-smooth 0.25 --close-pixels 1 --simplify 3 --vector-smooth 0 --min-region-area 35 --segmentation color --trace-style pixel --output output
if errorlevel 1 exit /b %errorlevel%
"%PY%" src\ai_verify_svg.py "%~1" "output\%STEM%\composition_entity.svg" --plan "output\%STEM%\entity_ai_plan.json" --output "output\%STEM%\composition_entity_verification.json" --model gpt-5.2
