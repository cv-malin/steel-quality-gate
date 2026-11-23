@echo off
chcp 65001 >nul
title Steel Quality Gate - 钢材表面缺陷智能质检台
cd /d "%~dp0"
set PY=D:\xtrain-venv\Scripts\python.exe
if not exist "%PY%" set PY=python
echo.
echo   Steel Quality Gate  v1.0.0
echo   Starting local demo server...
echo.
start "" http://127.0.0.1:8766
"%PY%" server.py
pause
