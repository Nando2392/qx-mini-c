@echo off
setlocal
if /I not "%~1"=="cuda" (
  echo Usage: build_cuda_msvc.bat cuda
  echo CUDA build is opt-in; use build_msvc.bat for the zero-CUDA CPU build.
  exit /b 2
)
set "CUDA_ROOT=%QX_CUDA_ROOT%"
if not defined CUDA_ROOT set "CUDA_ROOT=%CUDA_PATH%"
if not defined CUDA_ROOT (
  echo Missing CUDA root. Set QX_CUDA_ROOT or CUDA_PATH to a CUDA toolkit directory.
  exit /b 3
)
if not exist "%CUDA_ROOT%\bin\nvcc.exe" (
  echo Invalid CUDA root: missing %CUDA_ROOT%\bin\nvcc.exe
  exit /b 3
)
if not exist "%CUDA_ROOT%\lib\x64\cudart.lib" (
  echo Invalid CUDA root: missing %CUDA_ROOT%\lib\x64\cudart.lib
  exit /b 3
)
if not exist "%CUDA_ROOT%\include\cuda_runtime.h" (
  echo Invalid CUDA root: missing %CUDA_ROOT%\include\cuda_runtime.h
  exit /b 3
)
where cl >nul 2>nul
if errorlevel 1 (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
  if errorlevel 1 exit /b %errorlevel%
)
call build_msvc.bat
if errorlevel 1 exit /b %errorlevel%
"%CUDA_ROOT%\bin\nvcc.exe" -arch=sm_89 -cudart shared -Iinclude -c src\qx_cuda_final_head.cu -o build\qx_cuda_final_head.obj
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c17 /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /Iinclude src\qx_format.c src\qx_gguf.c src\qx_tokenizer.c src\qx_qxf_main.c build\qx_avx2.obj build\qx_cuda_final_head.obj /Fo:build\ /Fe:build\qxqxf_cuda.exe /link /Brepro /LIBPATH:"%CUDA_ROOT%\lib\x64" cudart.lib
exit /b %errorlevel%
