@echo off
pyinstaller camera_debugger.spec --clean --noconfirm
pyinstaller trainer.spec --clean --noconfirm
echo Build complete.
echo   camera_debugger: dist\camera_debugger\camera_debugger.exe
echo   trainer:         dist\trainer\trainer.exe
