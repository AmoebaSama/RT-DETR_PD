@echo off
setlocal
set "APP_DIR=%~dp0"
set "ROOT_DIR=%APP_DIR%.."
set "WORKSPACE_PY=%ROOT_DIR%\.venv\Scripts\python.exe"

if exist "%WORKSPACE_PY%" (
	"%WORKSPACE_PY%" "%APP_DIR%launch_rtdetr_ai.py"
	goto :eof
)

python "%APP_DIR%launch_rtdetr_ai.py"
endlocal