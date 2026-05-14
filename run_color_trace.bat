@echo off
if "%~1"=="" (
  echo Uso: run_color_trace.bat input\immagine.png
  exit /b 1
)
set PY=C:\Users\l.ercoli\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
"%PY%" src\color_trace_svg.py "%~1" --colors 10 --min-region-area 45 --simplify 1.0 --base-simplify 1.4 --curve-strength 0.32 --close 1 --open 0
