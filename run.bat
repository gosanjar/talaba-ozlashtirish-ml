@echo off
REM Talaba monitoringi: virtual muhit, kutubxonalar va server
cd /d "%~dp0"
if not exist .venv (
  echo Virtual muhit yaratilmoqda...
  python -m venv .venv
  .venv\Scripts\pip install -r requirements.txt
)
echo Server: http://localhost:8000
.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
