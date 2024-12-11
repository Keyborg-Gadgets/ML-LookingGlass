cmake_minimum_required(VERSION 3.14)

project(StandAloneCapture LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(WINDOWS_SDK_PATH "C:/Program Files (x86)/Windows Kits/10")

set_source_files_properties(DevicesAndShaders.c PROPERTIES LANGUAGE C)

include_directories(
    Onnx/include
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/ucrt
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/um
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/shared
)

add_library(DevicesAndShaders STATIC "DevicesAndShaders.c")
add_executable (StandAloneCapture "StandAloneCapture.cpp" 
                                  "StandAloneCapture.h"  
                                  "HwndsAndWindowManagement.h" 
                                  "pch.h" 
                                  "globals.h" 
                                  "DevicesAndShaders.h" 
                                  "DevicesAndShaderscpp.h" 
                                  "Onnx.h")

target_link_libraries(${PROJECT_NAME} PRIVATE
    d3d11
    dxgi
    dwmapi
    winmm
    dcomp
    user32
    d3dcompiler
    dxguid
    ntdll
    d2d1
    DevicesAndShaders
    dxguid 
    uuid 
    user32 
    advapi32 
    d3dcompiler
    ${PROJECT_SOURCE_DIR}/Onnx/lib/onnxruntime.lib
)

