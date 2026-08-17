@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
if not defined LLAMA_CPP_DIR set LLAMA_CPP_DIR=%~dp0..\..\llama.cpp-k3
set LLAMA=%LLAMA_CPP_DIR%
for /f %%i in ('git -C "%LLAMA%" rev-parse HEAD') do set LLAMA_COMMIT=%%i
cl /nologo /std:c++17 /O2 /W4 /EHsc /D_CRT_SECURE_NO_WARNINGS /DLLAMA_COMMIT=\"%LLAMA_COMMIT%\" /I"%LLAMA%\include" /I"%LLAMA%\src" /I"%LLAMA%\ggml\include" tests\llama_reference_oracle.cpp /Fe:build\llama_reference_oracle.exe /link /LIBPATH:"%LLAMA%\build-k3-cpu\src\Release" /LIBPATH:"%LLAMA%\build-k3-cpu\ggml\src\Release" llama.lib ggml-cpu.lib ggml-base.lib ggml.lib
if errorlevel 1 exit /b %errorlevel%
for %%d in (llama.dll ggml.dll ggml-base.dll ggml-cpu.dll) do (
  copy /y "%LLAMA%\build-k3-cpu\bin\Release\%%d" build\ >nul
  if errorlevel 1 exit /b 1
)
exit /b %errorlevel%
