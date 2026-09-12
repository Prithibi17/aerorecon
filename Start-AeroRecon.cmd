@echo off
cd /d "%~dp0"
echo AeroRecon - open http://127.0.0.1:8765 in your browser.
echo Keep this window open while using the app. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m uvicorn pipeline.server:app --host 127.0.0.1 --port 8765
pause
