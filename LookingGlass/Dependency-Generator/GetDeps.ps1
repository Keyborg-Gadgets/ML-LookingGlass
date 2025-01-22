TASKKILL /F /IM msedge.exe
$possiblePaths = @(
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
)
$edgePath = $possiblePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edgePath) {
    Write-Error "Microsoft Edge not found."
    return
}

&"$edgePath" --app=https://developer.nvidia.com/login --remote-debugging-port=9223 --disable-extensions

python.exe "$PSScriptRoot\deps.py"; TASKKILL /F /IM msedge.exe

$harFilePath = Join-Path -Path $PSScriptRoot -ChildPath "network_requests.har"

$harContent = Get-Content -Path $harFilePath -Raw

$regex = "https:\/\/developer\.download\.nvidia\.com\/compute\/machine-learning\/tensorrt\/secure\/\d+\.\d+\.\d+\/zip\/TensorRT-\d+\.\d+\.\d+\.\d+\.Windows10\.x86_64\.cuda-\d+\.\d+\.zip\?__token__=exp=\d+~hmac=[\da-f]+&t=[^""\s]+"

$match = [regex]::Match($harContent, $regex)

if ($match.Success) {
    $url = $match.Value
    $destinationPath = Join-Path -Path $PSScriptRoot -ChildPath "TensorRT.zip"

    if (-not (Test-Path -Path $destinationPath)) {
        # Suppress progress bar
        $ProgressPreference = 'SilentlyContinue'

        Clear-Host 
        Write-Output "Downloading TensorRT, The browser window was just for the auth token."

        # Download the file
        Invoke-WebRequest -Uri $url -OutFile $destinationPath -UseBasicParsing -Headers @{ "User-Agent" = "Mozilla/5.0" }

        Write-Output "Downloaded to: $destinationPath"
    } else {
        Write-Output "TensorRT.zip already exists at: $destinationPath"
    }
} else {
    Write-Output "No matching URLs found."
}

Write-Output "Downloading Cudnn"
$destinationPath = Join-Path -Path $PSScriptRoot -ChildPath "cudnn.zip"
if (-not (Test-Path -Path $destinationPath)) {
    Invoke-WebRequest -Uri "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.8.0.121_cuda12-archive.zip" -OutFile $destinationPath -UseBasicParsing -Headers @{ "User-Agent" = "Mozilla/5.0" }
    Write-Output "Downloaded to: $destinationPath"
} else {
    Write-Output "cudnn.zip already exists at: $destinationPath"
}

Write-Output "Downloading Cuda Installer"
$destinationPath = Join-Path -Path $PSScriptRoot -ChildPath "cuda12.3.exe"
if (-not (Test-Path -Path $destinationPath)) {
    Invoke-WebRequest -Uri "https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_546.12_windows.exe" -OutFile $destinationPath -UseBasicParsing -Headers @{ "User-Agent" = "Mozilla/5.0" }
    Write-Output "Downloaded to: $destinationPath"
} else {
    Write-Output "cuda12.3.exe already exists at: $destinationPath"
}

Write-Output "Downloading Cuda Setup Injector"
$destinationPath = Join-Path -Path $PSScriptRoot -ChildPath "injector.exe"
if (-not (Test-Path -Path $destinationPath)) {
    Invoke-WebRequest -Uri "https://github.com/Keyborg-Gadgets/CUDA-Setup-Injector/releases/download/v0.9.0/Injector.exe" -OutFile $destinationPath -UseBasicParsing -Headers @{ "User-Agent" = "Mozilla/5.0" }
    Write-Output "Downloaded to: $destinationPath"
} else {
    Write-Output "injector.exe already exists at: $destinationPath"
}

$exclude = Join-Path -Path $PSScriptRoot -ChildPath "Injector.exe"
Add-MpPreference -ExclusionPath "$exclude"

Write-Output "Installing Cuda Dependencies"
#TASKKILL /F /IM devenv.exe
& "$PSScriptRoot\cuda12.3.exe" -s cudart_12.3 nvcc_12.3 cufft_12.3 cufft_dev_12.3 curand_12.3 curand_dev_12.3 cusolver_12.3 cusolver_dev_12.3 cusparse_12.3 cusparse_dev_12.3 cublas_12.3 cublas_dev_12.3
$injectorPath = Join-Path -Path $PSScriptRoot -ChildPath "injector.exe"
function Check-Process {
    $processName = "setup"
    while ($true) {
        $process = Get-Process -Name $processName -ErrorAction SilentlyContinue
        if ($process) {
            Write-Output "$processName detected. Launching injector.exe... https://github.com/Keyborg-Gadgets/CUDA-Setup-Injector"
            Start-Process -FilePath $injectorPath
            break
        }
    }
}
Check-Process

$directoryPath = Join-Path -Path $PSScriptRoot -ChildPath "CUDA\v12.3"
if (-not (Test-Path -Path $directoryPath -PathType Container)) {
    # Create the directory if it does not exist
    New-Item -Path $directoryPath -ItemType Directory
    Write-Output "Directory created: $directoryPath"
} else {
    Write-Output "Directory already exists: $directoryPath"
}

$processName = "setup"

while ($true) {
    $process = Get-Process -Name $processName -ErrorAction SilentlyContinue
    if (-not $process) {
        break
    }
    Start-Sleep -Seconds 1
}

$folders = Get-ChildItem -Path $directoryPath -Directory | Select-Object -ExpandProperty Name

$cudnnZipPath = Join-Path -Path $PSScriptRoot -ChildPath "cudnn.zip"
$tensorRTZipPath = Join-Path -Path $PSScriptRoot -ChildPath "TensorRT.zip"

function Copy-ZipFilesToMatchingFolders {
    param (
        [string]$zipPath,
        [string[]]$folders,
        [string]$destinationPath
    )

    $tempDir = New-TemporaryFile
    Remove-Item $tempDir
    New-Item -Path $tempDir -ItemType Directory

    Write-Output "Extracting $zipPath to $tempDir"
    Expand-Archive -Path $zipPath -DestinationPath $tempDir

    # Automatically descend into the first folder inside the extracted directory
    $extractedRoot = Get-ChildItem -Path $tempDir | Where-Object { $_.PSIsContainer } | Select-Object -First 1
    if ($extractedRoot -eq $null) {
        Write-Error "No folders found inside the extracted archive."
        return
    }

    $basePath = $extractedRoot.FullName

    foreach ($folder in $folders) {
        $sourcePath = Join-Path -Path $basePath -ChildPath $folder
        $destPath = Join-Path -Path $destinationPath -ChildPath $folder
        if (Test-Path -Path $sourcePath -PathType Container) {
            Write-Output "Copying files from $sourcePath to $destPath"
            Copy-Item -Path $sourcePath\* -Destination $destPath -Recurse -Force
        }
    }

    Remove-Item -Path $tempDir -Recurse -Force
}

Copy-ZipFilesToMatchingFolders -zipPath $cudnnZipPath -folders $folders -destinationPath $directoryPath

Copy-ZipFilesToMatchingFolders -zipPath $tensorRTZipPath -folders $folders -destinationPath $directoryPath

# Move all DLLs to the bin folder
$binPath = "$PSScriptRoot\CUDA\v12.3\bin"
if (-not (Test-Path -Path $binPath -PathType Container)) {
    # Create the bin directory if it does not exist
    New-Item -Path $binPath -ItemType Directory
    Write-Output "Bin directory created: $binPath"
} else {
    Write-Output "Bin directory already exists: $binPath"
}

# Move DLL files to the bin directory
$allDllFiles = Get-ChildItem -Path "$PSScriptRoot\CUDA\v12.3" -Recurse -Filter *.dll
foreach ($dllFile in $allDllFiles) {
    $destination = Join-Path -Path $binPath -ChildPath $dllFile.Name
    Move-Item -Path $dllFile.FullName -Destination $destination -Force
    Write-Output "Moved $($dllFile.FullName) to $destination"
}

$cudaDir = "$PSScriptRoot\CUDA"
$hardLinkPath = "$PSScriptRoot\..\CUDA"
if (-not (Test-Path -Path $hardLinkPath)) {
    cmd /c mklink /J "$hardLinkPath" "$cudaDir"
    Write-Output "Hard link created: $hardLinkPath -> $cudaDir"
} else {
    Write-Output "Hard link already exists: $hardLinkPath"
}

Write-Output "Files copied successfully."

