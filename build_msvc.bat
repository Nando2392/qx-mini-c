@echo off
where cl >nul 2>nul
if errorlevel 1 (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
  if errorlevel 1 exit /b %errorlevel%
)
if not exist build mkdir build
cl /nologo /std:c17 /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /Iinclude src\qxfit.c src\qx_main.c /Fo:build\ /Fe:build\qxfit.exe
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c17 /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /Iinclude src\qx_format.c src\qx_gguf.c src\qx_qxf_main.c /Fo:build\ /Fe:build\qxqxf.exe
exit /b %errorlevel%
