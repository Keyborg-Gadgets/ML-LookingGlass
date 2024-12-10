cmake_minimum_required(VERSION 3.14)

project(StandAloneCapture LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(_AMD64_)

set(WINDOWS_SDK_PATH "C:/Program Files (x86)/Windows Kits/10")
set(VCPKG_INSTALLATION_ROOT "C:/Repos/vcpkg")
include("${VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake")

set_source_files_properties(DevicesAndShaders.c PROPERTIES LANGUAGE C)

include_directories(
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/ucrt
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/um
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/shared
    ${VCPKG_INSTALLATION_ROOT}/installed/x64-windows/include
)

add_library(DevicesAndShaders STATIC "DevicesAndShaders.c")
add_executable (StandAloneCapture "StandAloneCapture.cpp" "StandAloneCapture.h"  "HwndsAndWindowManagement.h" "pch.h" "globals.h" "DevicesAndShaders.h" "DevicesAndShaderscpp.h")

target_precompile_headers(StandAloneCapture PRIVATE 
<d3d11.h>
<Windows.Graphics.Capture.Interop.h>
<windows.graphics.directx.direct3d11.interop.h>
<winrt/Windows.Foundation.Metadata.h>
<winrt/Windows.Graphics.Capture.h>
<winrt/Windows.System.h>
)

find_package(cppwinrt CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
    d3d11
    dxgi
    windowsapp
    dwmapi
    winmm
    dcomp
    user32
    d3dcompiler
    dxguid
    ntdll
    d2d1
    DevicesAndShaders
    d3d10_1 
    dxguid 
    uuid 
    gdi32 
    user32 
    advapi32 
    d3dcompiler
)

