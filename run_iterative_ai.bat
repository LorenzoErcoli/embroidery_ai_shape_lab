@echo off
if "%~1"=="" (
  echo Uso: run_iterative_ai.bat input\immagine.png
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
"%PY%" src\iterate_ai_pipeline.py "%~1" --max-attempts 3 --target-score 60 --model gpt-5.2

