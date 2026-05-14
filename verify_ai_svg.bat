@echo off
if "%~2"=="" (
  echo Uso: verify_ai_svg.bat input\immagine.png output\nome\composition_ai.svg [output\nome\ai_plan.json]
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
if "%~3"=="" (
  "%PY%" src\ai_verify_svg.py "%~1" "%~2" --model gpt-5.2
) else (
  "%PY%" src\ai_verify_svg.py "%~1" "%~2" --plan "%~3" --model gpt-5.2
)

