@echo off
REM usage: train.bat [--windows 200 300 500] [--families A_feature B_emg_cnn C_imu_cnn D_twostream D_twostream_mt] [--split trial^|session^|participant] [--sessions DIR ...]
cd /d "%~dp0"
".venv\Scripts\python.exe" -W ignore -m emgmonitor.trainer %*
