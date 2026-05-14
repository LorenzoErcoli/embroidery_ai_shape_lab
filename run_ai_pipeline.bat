@echo off
if "%~1"=="" (
  echo Uso: run_ai_pipeline.bat input\immagine.png
  echo Richiede OPENAI_API_KEY nell'ambiente.
  exit /b 1
)
if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if "%%A"=="OPENAI_API_KEY" set OPENAI_API_KEY=%%B
  )
)
if "%OPENAI_API_KEY%"=="" (
  echo Manca OPENAI_API_KEY. Inseriscila nel file .env
  exit /b 1
)
set PY=C:\Users\l.ercoli\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
for %%F in ("%~1") do set STEM=%%~nF
"%PY%" src\ai_embroidery_plan.py "%~1" --model gpt-5.2
if errorlevel 1 exit /b %errorlevel%
"%PY%" src\ai_plan_to_svg.py "%~1" "output\%STEM%\ai_plan.json" --color-tolerance 52 --min-region-area 160 --simplify 6 --edge-smooth 1.4 --close-pixels 2 --overlap-mode details-only --shadow-mode force --shadow-min-luma-gap 14 --shape-prior auto --max-overlay-components 0 --max-detail-components 0 --max-outline-components 0
