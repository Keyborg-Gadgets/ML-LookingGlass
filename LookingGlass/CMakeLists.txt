﻿cmake_minimum_required(VERSION 3.10)

project(LookingGlass LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(WINDOWS_SDK_PATH "C:/Program Files (x86)/Windows Kits/10")

set_source_files_properties(DevicesAndShaders.c PROPERTIES LANGUAGE C)

include_directories(
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/ucrt
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/um
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/shared
    ${PROJECT_SOURCE_DIR}/CUDA/v12.3/include
)

link_directories(
    ${PROJECT_SOURCE_DIR}/CUDA/v12.3/lib/x64
    ${PROJECT_SOURCE_DIR}/CUDA/v12.3/lib
)

add_library(DevicesAndShaders STATIC "DevicesAndShaders.c" "orbd.h")
add_executable (${PROJECT_NAME} "LookingGlass.cpp" 
                                  "LookingGlass.h"  
                                  "HwndsAndWindowManagement.h" 
                                  "pch.h" 
                                  "globals.h" 
                                  "DevicesAndShaders.h" 
                                  "DevicesAndShaderscpp.h" 
                                  "cudaFunctions.cuh")


target_link_libraries(${PROJECT_NAME} PRIVATE 
    DevicesAndShaders
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
    dxguid 
    uuid 
    user32 
    advapi32 
    d3dcompiler
    cuda
    cudart
    cufft
    curand
    cublas
    cudnn
    nvinfer
    nvonnxparser 
)

set(POWERSHELL_SCRIPT  ${CMAKE_CURRENT_SOURCE_DIR}/SetStaticBinaryResources.ps1)

add_custom_target(run_powershell_script ALL
    COMMAND powershell -ExecutionPolicy Bypass -File ${POWERSHELL_SCRIPT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running PowerShell script after build..."
)
add_dependencies(${PROJECT_NAME} run_powershell_script)