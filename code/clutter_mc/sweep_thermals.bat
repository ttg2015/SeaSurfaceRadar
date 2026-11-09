@echo off
rem Live thermal/load graph of the sweep (refreshes every 30 s). Optional arg: hours to show (0 = all).
cd /d "%~dp0"
start "" pythonw thermal_view.py %1
