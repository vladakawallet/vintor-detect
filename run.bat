@echo off
title vintor-detect

setlocal EnableDelayedExpansion

REM --- 1. Загрузка переменных из .env ---
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    set %%A=%%B
)

"%PYTHON%" -m pip install --upgrade pip

"%PYTHON%" -m pip install -r requirements.txt



uvicorn api:app --host 0.0.0.0 --port 8000

REM --- 4. Оставить окно консоли открытым для отладки ---
pause
