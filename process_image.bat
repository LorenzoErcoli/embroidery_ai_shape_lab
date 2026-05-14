@echo off
if "%~1"=="" (
  echo Uso: process_image.bat input\immagine.png
  exit /b 1
)
set PY=C:\Users\l.ercoli\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
"%PY%" src\image_shape_lab.py "%~1" --colors 6 --bg-tolerance 35 --min-region-area 80 --simplify 3

