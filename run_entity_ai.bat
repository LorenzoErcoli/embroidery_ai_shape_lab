@echo off
if "%~1"=="" (
  echo Uso: run_entity_ai.bat input\immagine.png
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
"%PY%" src\ai_entity_plan.py "%~1" --model gpt-5.2
if errorlevel 1 exit /b %errorlevel%
"%PY%" src\ai_verify_entity_plan.py "%~1" "output\%STEM%\entity_ai_plan.json" --output "output\%STEM%\entity_plan_verification.json" --model gpt-5.2
if errorlevel 1 exit /b %errorlevel%
"%PY%" src\entity_plan_to_svg.py "%~1" "output\%STEM%\entity_ai_plan.json" --color-tolerance 58 --edge-smooth 1.2 --close-pixels 4 --simplify 5 --vector-smooth 1 --min-region-area 180 --segmentation auto --sam-checkpoint "models\sam_vit_b_01ec64.pth"
if errorlevel 1 exit /b %errorlevel%
"%PY%" src\ai_verify_svg.py "%~1" "output\%STEM%\composition_entity.svg" --plan "output\%STEM%\entity_ai_plan.json" --output "output\%STEM%\composition_entity_verification.json" --model gpt-5.2
