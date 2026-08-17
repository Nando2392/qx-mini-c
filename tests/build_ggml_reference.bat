@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
if not defined LLAMA_CPP_DIR set LLAMA_CPP_DIR=%~dp0..\..\llama.cpp-k3
set LLAMA=%LLAMA_CPP_DIR%
cl /nologo /std:c17 /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /I"%LLAMA%\ggml\src" /I"%LLAMA%\ggml\include" tests\ggml_reference_decode.c /Fe:build\ggml_reference_decode.exe /link "%LLAMA%\build-k3-cpu\ggml\src\Release\ggml-cpu.lib" "%LLAMA%\build-k3-cpu\ggml\src\Release\ggml-base.lib" "%LLAMA%\build-k3-cpu\ggml\src\Release\ggml.lib"
exit /b %errorlevel%
