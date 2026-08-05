@echo off
set BACKUP=build\userdata_backup

REM ===== 备份用户数据（构建会重建 dist 目录，先备份再恢复）=====
if exist dist\trainer\dataset xcopy /E /I /Y dist\trainer\dataset %BACKUP%\trainer_dataset >nul
if exist dist\trainer\*.pt copy /Y dist\trainer\*.pt %BACKUP%\ >nul
if exist dist\camera_debugger\*.pt copy /Y dist\camera_debugger\*.pt %BACKUP%\ >nul

pyinstaller camera_debugger.spec --clean --noconfirm
pyinstaller trainer.spec --clean --noconfirm

REM ===== 恢复用户数据 =====
if exist %BACKUP%\trainer_dataset xcopy /E /I /Y %BACKUP%\trainer_dataset dist\trainer\dataset >nul
if exist %BACKUP%\*.pt copy /Y %BACKUP%\*.pt dist\trainer\ >nul

echo Build complete.
echo   camera_debugger: dist\camera_debugger\camera_debugger.exe
echo   trainer:         dist\trainer\trainer.exe
