<#
.SYNOPSIS
  PowerShell script to locate cl.exe for x64, compile CUDA objects/libraries,
  generate a TensorRT engine (if needed), and set up the output directories.

.DESCRIPTION
  1. Searches Visual Studio paths for x64 cl.exe.
  2. Adds cl.exe directory to PATH (current session).
  3. Compiles CUDA source to both Release and Debug libraries.
  4. Moves library files to a specified CUDA lib folder.
  5. Generates a TensorRT engine file (if it doesn’t already exist).
  6. Copies necessary DLLs and the engine file into specified output directories.
#>

#=== PARAMETERS & PATHS ========================================================

# Define the search paths for Visual Studio cl.exe
$searchPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC"
)

# CUDA directories
$cudaBin  = "$PSScriptRoot\CUDA\v12.3\bin"
$cudaLib  = "$PSScriptRoot\CUDA\v12.3\lib"

# Output library and object filenames
$releaseObj =  "$PSScriptRoot\cudaFunctions.obj"
$releaseLib =  "$PSScriptRoot\cudaFunctions.lib"
$debugObj   =  "$PSScriptRoot\cudaFunctions_debug.obj"
$debugLib   =  "$PSScriptRoot\cudaFunctions_debug.lib"

# Final destination of the compiled libraries
$releaseLibDest = Join-Path $cudaLib "cudaFunctions.lib"
$debugLibDest   = Join-Path $cudaLib "cudaFunctions_debug.lib"

# Model/engine paths
$modelDir   = "$PSScriptRoot\..\Model"
$onnxModel  = Join-Path $modelDir "modified_out.sim.onnx"
$engineFile = Join-Path $modelDir "rtdetr_r18vd_6x_coco-fp16.engine"

# Output directories
$targetDir1 = "$PSScriptRoot\..\out\build\x64-debug\LookingGlass"
$targetDir2 = "$PSScriptRoot\..\out\build\x64-release\LookingGlass"

#=== FUNCTIONS ================================================================

function Find-CLExe {
    param (
        [string[]]$Paths
    )
    $clExeFiles = @()
    foreach ($path in $Paths) {
        $clExeFiles += Get-ChildItem -Path $path -Filter cl.exe -Recurse -ErrorAction SilentlyContinue
    }
    return $clExeFiles
}

#=== SEARCH FOR CL.EXE & ADD TO PATH ==========================================
$clExeFiles = Find-CLExe -Paths $searchPaths
$clExeFile  = $clExeFiles | Where-Object { $_.FullName -match "x64" } | Select-Object -First 1

if ($clExeFile) {
    Write-Host "First x64 cl.exe found at: $($clExeFile.FullName)"
    $clExeDir = Split-Path -Path $clExeFile.FullName
    $env:PATH = "$clExeDir;$env:PATH"
    Write-Host "Added $clExeDir to PATH for this session."
}
else {
    Write-Warning "x64 cl.exe not found in the specified locations."
}

#=== COMPILE CUDA SOURCES =====================================================

# Release Build
& "$cudaBin\nvcc.exe" -allow-unsupported-compiler -c -o $releaseObj cudaFunctions.cu `
    -Xcompiler "/MD /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
& "$cudaBin\nvcc.exe" -lib -o $releaseLib $releaseObj

# Debug Build
& "$cudaBin\nvcc.exe" -allow-unsupported-compiler -c -o $debugObj cudaFunctions.cu `
    -Xcompiler "/MDd /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH /Zi /Od"
& "$cudaBin\nvcc.exe" -lib -o $debugLib $debugObj

#=== MOVE & CLEANUP ===========================================================

Move-Item -Path $releaseLib -Destination $releaseLibDest -Force
Move-Item -Path $debugLib   -Destination $debugLibDest   -Force

Remove-Item -Path $releaseObj -Force -ErrorAction SilentlyContinue
Remove-Item -Path $debugObj   -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$PSScriptRoot\cudaFunctions.exp" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$PSScriptRoot\vc140.pdb"         -Force -ErrorAction SilentlyContinue

#=== PREPARE OUTPUT DIRECTORIES ===============================================

foreach ($dir in @($targetDir1, $targetDir2)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

#=== SET ENVIRONMENT VARIABLES ================================================

$env:Path                  = "$cudaBin;$env:Path"
$env:Path                  = "$cudaLib;$env:Path"
$env:Path                  = "$cudaLib\x64;$env:Path"
$env:CUDA_TOOLKIT_ROOT_DIR = "$PSScriptRoot\CUDA\v12.3"

#=== GENERATE ENGINE (IF NEEDED) ==============================================

Set-Location $cudaBin

#if (-not (Test-Path $engineFile)) {
#    Write-Host "Engine file not found. Generating engine from ONNX model..."
#    trtexec --onnx=$onnxModel --saveEngine=$engineFile --fp16
#    if ($LASTEXITCODE -ne 0) {
#        Write-Error "Failed to generate engine file."
#        Set-Location $PSScriptRoot
#        exit $LASTEXITCODE
#    }
#}

#=== SET UP HARD LINKS FOR DLLs & COPY ENGINE FILE ============================

$nvinferDll                = Join-Path $cudaBin "nvinfer.dll"
$cudartDll                 = Join-Path $cudaBin "cudart64_12.dll"
$nvonnxParserDll           = Join-Path $cudaBin "nvonnxparser.dll"
$nvinferPluginDll          = Join-Path $cudaBin "nvinfer_plugin.dll"
$nvinferBuilderResourceDll = Join-Path $cudaBin "nvinfer_builder_resource.dll"

# Adjust file names/paths as necessary for your actual cublas/cudnn versions.
$cublasDll    = Join-Path $cudaBin "cublas64_12.dll"
$cublasLtDll  = Join-Path $cudaBin "cublasLt64_12.dll"
$cudnnDll     = Join-Path $cudaBin "cudnn64_8.dll"

foreach ($dir in @($targetDir1, $targetDir2)) {
    New-Item -ItemType HardLink -Path (Join-Path $dir "nvinfer.dll")                   -Target $nvinferDll                -Force | Out-Null
    New-Item -ItemType HardLink -Path (Join-Path $dir "cudart64_12.dll")               -Target $cudartDll                 -Force | Out-Null
    New-Item -ItemType HardLink -Path (Join-Path $dir "nvonnxparser.dll")              -Target $nvonnxParserDll           -Force | Out-Null
    New-Item -ItemType HardLink -Path (Join-Path $dir "nvinfer_plugin.dll")            -Target $nvinferPluginDll          -Force | Out-Null
    New-Item -ItemType HardLink -Path (Join-Path $dir "nvinfer_builder_resource.dll")  -Target $nvinferBuilderResourceDll -Force | Out-Null

    New-Item -ItemType HardLink -Path (Join-Path $dir "cublas64_12.dll")               -Target $cublasDll    -Force | Out-Null
    New-Item -ItemType HardLink -Path (Join-Path $dir "cublasLt64_12.dll")             -Target $cublasLtDll  -Force | Out-Null
    New-Item -ItemType HardLink -Path (Join-Path $dir "cudnn64_8.dll")                 -Target $cudnnDll     -Force | Out-Null
    
    Copy-Item -Path $engineFile -Destination $dir -Force
}

Write-Host "Engine file copied to both target directories."
Write-Host "Setup complete."

Set-Location $PSScriptRoot
