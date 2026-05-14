@echo off
if "%~1"=="" (
  echo Uso: run_ai_layered_trace.bat input\immagine.png
  echo Richiede output\nome\entity_ai_plan.json gia' generato.
  exit /b 1
)
set PY=C:\Users\l.ercoli\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
for %%F in ("%~1") do set STEM=%%~nF
"%PY%" src\ai_layered_color_trace.py "%~1" "output\%STEM%\entity_ai_plan.json"
