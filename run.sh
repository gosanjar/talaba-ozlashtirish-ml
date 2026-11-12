#!/usr/bin/env bash
# Talaba monitoringi: virtual muhit yaratadi, kutubxonalarni o'rnatadi va serverni ishga tushiradi.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "Virtual muhit yaratilmoqda…"
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip -q
  .venv/bin/pip install -r requirements.txt
fi
PORT=${PORT:-8000}
echo "Server: http://localhost:$PORT  (birinchi ishga tushishda model ~1 daqiqa o'qitiladi)"
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
