@echo off
if "%~1"=="" (
  echo Uso: run_spatial_ai.bat input\immagine.png
  echo Richiede OPENAI_API_KEY nel file .env o nell'ambiente.
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
"%PY%" src\ai_spatial_plan.py "%~1" --model gpt-5.2
if errorlevel 1 exit /b %errorlevel%
"%PY%" src\spatial_plan_to_svg.py "%~1" "output\%STEM%\spatial_ai_plan.json" --color-tolerance 58 --edge-smooth 1.2 --close-pixels 3 --simplify 7 --min-region-area 160
if errorlevel 1 exit /b %errorlevel%
"%PY%" src\ai_verify_svg.py "%~1" "output\%STEM%\composition_spatial.svg" --plan "output\%STEM%\spatial_ai_plan.json" --output "output\%STEM%\composition_spatial_verification.json" --model gpt-5.2

