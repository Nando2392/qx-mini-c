@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0.."
set "OUT=build\issue86-driver-testonly"
if not exist "%OUT%\obj" mkdir "%OUT%\obj"
cl /nologo /std:c17 /O2 /W4 /arch:AVX2 /D_CRT_SECURE_NO_WARNINGS /Iinclude /c src\qx_avx2.c /Fo:"%OUT%\obj\qx_avx2.obj"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c17 /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc /c src\qx_format.c /Fo:"%OUT%\obj\qx_format.obj"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c17 /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc /c src\qx_expert_cache.c /Fo:"%OUT%\obj\qx_expert_cache.obj"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c17 /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc /c src\qx_gguf.c /Fo:"%OUT%\obj\qx_gguf.obj"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c17 /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc /c src\qx_tokenizer.c /Fo:"%OUT%\obj\qx_tokenizer.obj"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c17 /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc /c src\qx_cuda_final_head_stub.c /Fo:"%OUT%\obj\qx_cuda_final_head_stub.obj"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c17 /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc /c tests\expert_cache_acceptance_driver.c /Fo:"%OUT%\obj\expert_cache_acceptance_driver.obj"
if errorlevel 1 exit /b %errorlevel%
link /nologo /Brepro /OUT:"%OUT%\expert_cache_acceptance_driver.exe" "%OUT%\obj\expert_cache_acceptance_driver.obj" "%OUT%\obj\qx_format.obj" "%OUT%\obj\qx_expert_cache.obj" "%OUT%\obj\qx_gguf.obj" "%OUT%\obj\qx_tokenizer.obj" "%OUT%\obj\qx_cuda_final_head_stub.obj" "%OUT%\obj\qx_avx2.obj"
exit /b %errorlevel%
