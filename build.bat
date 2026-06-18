@echo off
pyinstaller camera_debugger.spec --clean --noconfirm
echo Build complete. Output: dist\camera_debugger\camera_debugger.exe
