cmake_minimum_required(VERSION 3.14)

project(LookingGlass LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(WINDOWS_SDK_PATH "C:/Program Files (x86)/Windows Kits/10")

set_source_files_properties(DevicesAndShaders.c PROPERTIES LANGUAGE C)
set_source_files_properties(cudaFunctions.cu PROPERTIES LANGUAGE CUDA) 

add_library(cudaFunctions STATIC cudaFunctions.cu)

if(MSVC) 
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>") 
endif()

include_directories(
    Onnx/include
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/ucrt
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/um
    ${WINDOWS_SDK_PATH}/Include/10.0.26100.0/shared
    ${PROJECT_SOURCE_DIR}/Onnx/include
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include"
)

link_directories(
    ${PROJECT_SOURCE_DIR}/Onnx/lib
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/lib/x64"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/lib"
)

add_library(DevicesAndShaders STATIC "DevicesAndShaders.c")
add_executable (${PROJECT_NAME} "LookingGlass.cpp" 
                                  "LookingGlass.h"  
                                  "HwndsAndWindowManagement.h" 
                                  "pch.h" 
                                  "globals.h" 
                                  "DevicesAndShaders.h" 
                                  "DevicesAndShaderscpp.h" 
                                  "Onnx.h"
                                  "cudaFunctions.cuh")

target_link_libraries(${PROJECT_NAME} PRIVATE 
    DevicesAndShaders
    cudaFunctions
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
    onnxruntime
    onnxruntime_providers_shared
    onnxruntime_providers_cuda
    cuda
    cudart
    cufft
    curand
    cublas
    cudnn
)

set(POWERSHELL_SCRIPT  ${CMAKE_CURRENT_SOURCE_DIR}/SetStaticBinaryResources.ps1)

add_custom_target(run_powershell_script ALL
    COMMAND powershell -ExecutionPolicy Bypass -File ${POWERSHELL_SCRIPT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running PowerShell script after build..."
)

add_dependencies(${PROJECT_NAME} run_powershell_script)