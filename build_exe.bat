@echo off
REM ============================================================
REM  Build .exe del Aplicativo MEF Q4 (Trabajo de Tesis)
REM ============================================================
REM  Usa el venv del proyecto + PyInstaller + el spec build_exe.spec
REM  Produce:  dist\MEF_Q4\MEF_Q4.exe  (carpeta completa)
REM            dist\MEF_Q4.zip          (lista para enviar)
REM ============================================================

setlocal

echo.
echo [1/4] Limpiando builds anteriores...
if exist build  rmdir /S /Q build
if exist dist   rmdir /S /Q dist
if exist MEF_Q4.zip del MEF_Q4.zip

echo.
echo [2/4] Empaquetando con PyInstaller...
.\venv\Scripts\pyinstaller.exe build_exe.spec --noconfirm
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller fallo. Revisa el log de arriba.
    exit /b 1
)

echo.
echo [3/4] Copiando README al paquete...
copy /Y RELEASE_README.txt dist\MEF_Q4\LEAME.txt > nul 2>&1

echo.
echo [4/4] Comprimiendo a ZIP...
powershell -Command "Compress-Archive -Path 'dist\MEF_Q4' -DestinationPath 'dist\MEF_Q4.zip' -Force"

echo.
echo ============================================================
echo  Build completo!
echo ============================================================
echo  Carpeta:  dist\MEF_Q4\         (todo lo que se instala)
echo  ZIP:      dist\MEF_Q4.zip      (envia este a tu profesor)
echo  Ejecutable: dist\MEF_Q4\MEF_Q4.exe
echo.
echo  Para probar:  dist\MEF_Q4\MEF_Q4.exe
echo ============================================================

endlocal
