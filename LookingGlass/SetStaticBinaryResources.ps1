# $sourceFile = "ComputeShader.hlsl"

# $targetDir1 = "../out/build/x64-debug/LookingGlass"
# $targetDir2 = "../out/build/x64-release/LookingGlass"

# $fullTargetDir1 = Resolve-Path -Path $targetDir1 -ErrorAction SilentlyContinue
# $fullTargetDir2 = Resolve-Path -Path $targetDir2 -ErrorAction SilentlyContinue

# if (-not $fullTargetDir1) {
#     New-Item -ItemType Directory -Path $targetDir1 -Force
# }

# if (-not $fullTargetDir2) {
#     New-Item -ItemType Directory -Path $targetDir2 -Force
# }

# Copy-Item -Path $sourceFile -Destination $targetDir1 -Force

# Copy-Item -Path $sourceFile -Destination $targetDir2 -Force

# Write-Host "Copied $sourceFile to $targetDir1 and $targetDir2"

# $sourceFile = "labels.txt"

# $targetDir1 = "../out/build/x64-debug/LookingGlass"
# $targetDir2 = "../out/build/x64-release/LookingGlass"

# $fullTargetDir1 = Resolve-Path -Path $targetDir1 -ErrorAction SilentlyContinue
# $fullTargetDir2 = Resolve-Path -Path $targetDir2 -ErrorAction SilentlyContinue

# if (-not $fullTargetDir1) {
#     New-Item -ItemType Directory -Path $targetDir1 -Force
# }

# if (-not $fullTargetDir2) {
#     New-Item -ItemType Directory -Path $targetDir2 -Force
# }

# Copy-Item -Path $sourceFile -Destination $targetDir1 -Force

# Copy-Item -Path $sourceFile -Destination $targetDir2 -Force

$targetDir1 = "$PSScriptRoot\..\out\build\x64-debug\LookingGlass"
$targetDir2 = "$PSScriptRoot/../out/build/x64-release/LookingGlass"
$modelDir = "$PSScriptRoot/../Model"
$onnxModel = "$modelDir/modified_out.sim.onnx"
$engineFile = "$modelDir/rtdetr_r18vd_6x_coco-fp16.engine"
$env:Path = "$PSScriptRoot\CUDA\v12.3\bin;$env:Path"
$env:Path = "$PSScriptRoot\CUDA\v12.3\lib;$env:Path"
$env:Path = "$PSScriptRoot\CUDA\v12.3\lib\x64;$env:Path"
$env:CUDA_TOOLKIT_ROOT_DIR = "$PSScriptRoot\CUDA\v12.3"


if (-Not (Test-Path $engineFile)) {
    Write-Host "Engine file not found. Generating engine file from ONNX model..."
    & $PSScriptRoot/CUDA/v12.3/bin/trtexec --onnx=$onnxModel --saveEngine=$engineFile --fp16
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to generate engine file."
        exit $LASTEXITCODE
    }
}

Copy-Item -Path $engineFile -Destination $targetDir1 -Force
Copy-Item -Path $engineFile -Destination $targetDir2 -Force

Write-Host "Engine file copied to target directories."

Write-Host "Copied $sourceFile to $targetDir1 and $targetDir2"