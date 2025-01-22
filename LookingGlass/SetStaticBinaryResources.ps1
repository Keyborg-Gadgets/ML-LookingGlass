$sourceFile = "ComputeShader.hlsl"

$targetDir1 = "../out/build/x64-debug/LookingGlass"
$targetDir2 = "../out/build/x64-release/LookingGlass"

$fullTargetDir1 = Resolve-Path -Path $targetDir1 -ErrorAction SilentlyContinue
$fullTargetDir2 = Resolve-Path -Path $targetDir2 -ErrorAction SilentlyContinue

if (-not $fullTargetDir1) {
    New-Item -ItemType Directory -Path $targetDir1 -Force
}

if (-not $fullTargetDir2) {
    New-Item -ItemType Directory -Path $targetDir2 -Force
}

Copy-Item -Path $sourceFile -Destination $targetDir1 -Force

Copy-Item -Path $sourceFile -Destination $targetDir2 -Force

Write-Host "Copied $sourceFile to $targetDir1 and $targetDir2"

$sourceFile = "labels.txt"

$targetDir1 = "../out/build/x64-debug/LookingGlass"
$targetDir2 = "../out/build/x64-release/LookingGlass"

$fullTargetDir1 = Resolve-Path -Path $targetDir1 -ErrorAction SilentlyContinue
$fullTargetDir2 = Resolve-Path -Path $targetDir2 -ErrorAction SilentlyContinue

if (-not $fullTargetDir1) {
    New-Item -ItemType Directory -Path $targetDir1 -Force
}

if (-not $fullTargetDir2) {
    New-Item -ItemType Directory -Path $targetDir2 -Force
}

Copy-Item -Path $sourceFile -Destination $targetDir1 -Force

Copy-Item -Path $sourceFile -Destination $targetDir2 -Force