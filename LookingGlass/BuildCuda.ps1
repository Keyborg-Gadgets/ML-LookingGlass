# Define the search paths
$searchPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC"
)

# Function to search for cl.exe
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

# Search for cl.exe in the defined paths
$clExeFiles = Find-CLExe -Paths $searchPaths

# Filter for the first x64 cl.exe
$clExeFile = $clExeFiles | Where-Object { $_.FullName -match "x64" } | Select-Object -First 1

# Display the result and add to PATH
if ($clExeFile) {
    Write-Host "First x64 cl.exe found at:"
    Write-Host $clExeFile.FullName

    # Get the directory containing cl.exe
    $clExeDir = Split-Path -Path $clExeFile.FullName

    # Add the directory to the PATH for this session
    $env:PATH = "$clExeDir;$env:PATH"
    Write-Host "Added $clExeDir to PATH for this session."
} else {
    Write-Host "x64 cl.exe not found in the specified locations."
}

&"$PSScriptRoot\CUDA\v12.3\bin\nvcc.exe" -allow-unsupported-compiler -c -o cudaFunctions.obj cudaFunctions.cu -Xcompiler "/MD /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
&"$PSScriptRoot\CUDA\v12.3\bin\nvcc.exe" -lib -o cudaFunctions.lib cudaFunctions.obj

$sourceLib = "$PSScriptRoot\cudaFunctions.lib"
$destinationLib = "$PSScriptRoot\CUDA\v12.3\lib\cudaFunctions.lib"
$objectFile = "$PSScriptRoot\cudaFunctions.obj"

Move-Item -Path $sourceLib -Destination $destinationLib -Force
Remove-Item -Path $objectFile -Force
Remove-Item -Path $PSScriptRoot\cudaFunctions.exp -Force


$targetDir1 = "$PSScriptRoot\..\out\build\x64-debug\LookingGlass"
$targetDir2 = "$PSScriptRoot/../out/build/x64-release/LookingGlass"
$modelDir = "$PSScriptRoot/../Model"
$onnxModel = "$modelDir/modified_out.sim.onnx"
$engineFile = "$modelDir/rtdetr_r18vd_6x_coco-fp16.engine"
$env:Path = "$PSScriptRoot\CUDA\v12.3\bin;$env:Path"
$env:Path = "$PSScriptRoot\CUDA\v12.3\lib;$env:Path"
$env:Path = "$PSScriptRoot\CUDA\v12.3\lib\x64;$env:Path"
$env:CUDA_TOOLKIT_ROOT_DIR = "$PSScriptRoot\CUDA\v12.3"

$targetDir1 = "$PSScriptRoot\..\out\build\x64-debug\LookingGlass"
$targetDir2 = "$PSScriptRoot/../out/build/x64-release/LookingGlass"

# Ensure target directories are created if they do not exist
if (-not (Test-Path $targetDir1)) {
    New-Item -ItemType Directory -Path $targetDir1 -Force
}

if (-not (Test-Path $targetDir2)) {
    New-Item -ItemType Directory -Path $targetDir2 -Force
}

Set-Location $PSScriptRoot/CUDA/v12.3/bin/

if (-Not (Test-Path $engineFile)) {
    Write-Host "Engine file not found. Generating engine file from ONNX model..."
    trtexec --onnx=$onnxModel --saveEngine=$engineFile --fp16
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to generate engine file."
        Set-Location $PSScriptRoot
        exit $LASTEXITCODE
    }
}

# Create hard links for nvinfer.dll and cudart64_12.dll
$nvinferDll = "$PSScriptRoot\CUDA\v12.3\bin\nvinfer.dll"
$cudartDll = "$PSScriptRoot\CUDA\v12.3\bin\cudart64_12.dll"

New-Item -ItemType HardLink -Path "$targetDir1\nvinfer.dll" -Target $nvinferDll -Force
New-Item -ItemType HardLink -Path "$targetDir1\cudart64_12.dll" -Target $cudartDll -Force
New-Item -ItemType HardLink -Path "$targetDir2\nvinfer.dll" -Target $nvinferDll -Force
New-Item -ItemType HardLink -Path "$targetDir2\cudart64_12.dll" -Target $cudartDll -Force

Copy-Item -Path $engineFile -Destination $targetDir1 -Force
Copy-Item -Path $engineFile -Destination $targetDir2 -Force

Write-Host "Engine file copied to target directories."

Write-Host "Copied $sourceFile to $targetDir1 and $targetDir2"

Set-Location $PSScriptRoot