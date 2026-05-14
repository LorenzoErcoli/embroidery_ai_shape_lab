@echo off
set PY=C:\Users\l.ercoli\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
"%PY%" src\make_sample_ball.py
"%PY%" src\image_shape_lab.py input\sample_ball.png --colors 6 --bg-tolerance 28 --min-region-area 120 --simplify 3

