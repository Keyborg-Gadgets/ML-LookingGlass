#pragma once
#include "pch.h"

#ifdef _DEBUG
uint32_t debug = 1;
#endif
#ifndef _DEBUG
uint32_t debug = 0;
#endif

float* rgbCudaArray;
std::string exeDir;
HRESULT hr = S_OK;
uint32_t imgsz = 740;
uint32_t enginesz = 640;
uint32_t xOfWindow = -1;
uint32_t yOfWindow = -1;
uint32_t xOfMouse = 0;
uint32_t yOfMouse = 0;
HHOOK hMouseHook = nullptr;
ULONG currentRes;
std::atomic<bool> winDone(false);

static bool previouslyInteractive = true;
bool windowAvailable = (xOfWindow != -1 && yOfWindow != -1);
bool inBounds = true;
bool interactive = true;

bool running = true;
HWND overlayHwnd;
HWND lookingGlassHwnd;
HMONITOR monitor;
HINSTANCE hInstance;
std::atomic<bool> texture_written = false;
int titleBarHeight;

uint32_t monitor_width = 0;
uint32_t monitor_height = 0;

IDXGIAdapter1* adapter = nullptr;

IDXGIFactory2* d3dfactory = nullptr;
ID2D1Factory2* d2dFactory = nullptr;
ID2D1Device1* d2dDevice = nullptr;
ID2D1DeviceContext* d2dContext = nullptr;

IDXGIDevice1* dxgiDevice = nullptr;
ID3D11RenderTargetView* renderTargetView = nullptr;

IDCompositionDevice* dcompDevice = nullptr;
IDCompositionTarget* dcompTarget = nullptr;
IDCompositionVisual* dcompVisual = nullptr;

IDXGISwapChain1* swapchain = nullptr;

IDXGISwapChain1* cudaTextureSwapchain = nullptr;
ID3D11Texture2D* cudaTexture = nullptr;
IDXGISurface2* cudaTextureBackBuffer = nullptr;
ID3D11UnorderedAccessView* cudaTextureUAV = nullptr;

IDXGISwapChain1* outputTextureSwapchain = nullptr;
ID3D11Texture2D* outputTexture = nullptr;
IDXGISurface2* outputTextureBackBuffer = nullptr;
ID3D11UnorderedAccessView* outputTextureUAV = nullptr;

ID3D11Device1* d3dDevice = nullptr;
ID3D11DeviceContext1* d3dContext = nullptr;

IDXGISurface2* dxgiBackBuffer = nullptr;
ID2D1Bitmap1* d2dBitmapBackBuffer = nullptr;
ID3D11Texture2D* d2dTextureBackBuffer = nullptr;
ID3D11Texture2D* desktopTexture = nullptr;

ID3D11ShaderResourceView* desktopShaderResourceView = nullptr;
IDXGIOutputDuplication* outputDuplication = nullptr;

ID3D11ComputeShader* ScanComputeShader = nullptr;
ID3D11ComputeShader* CopyComputeShader = nullptr;

ID3D11Buffer* xBuffer = nullptr;
ID3D11Buffer* yBuffer = nullptr;
ID3D11UnorderedAccessView* xUAV = nullptr;
ID3D11UnorderedAccessView* yUAV = nullptr;
ID3D11Buffer* xReadback = nullptr;
ID3D11Buffer* yReadback = nullptr;
ID3D11Buffer* regionXBuffer = nullptr;
ID3D11Buffer* regionYBuffer = nullptr;
ID3D11UnorderedAccessView* regionXUAV = nullptr;
ID3D11UnorderedAccessView* regionYUAV = nullptr;
ID3D11Buffer* debugBuffer = nullptr;
ID3D11UnorderedAccessView* debugUAV = nullptr;

ID3D11SamplerState* samplerState = nullptr;

cudaGraphicsResource* cudaResource = nullptr;
cudaArray_t cudaTextureArray;
float* d_output0 = nullptr;
float* d_output1 = nullptr;

inline std::string getExecutableDirectory() {
    char path[MAX_PATH];
    DWORD length = GetModuleFileNameA(NULL, path, MAX_PATH);
    if (length == 0) {
        return "";
    }
    std::string fullPath(path, length);
    size_t pos = fullPath.find_last_of("\\/");
    return (pos == std::string::npos) ? "" : fullPath.substr(0, pos);
}

class Timer {
public:
    Timer() : last_time(std::chrono::high_resolution_clock::now()) {}

    double sinceLast() {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = current_time - last_time;
        last_time = current_time;
        std::cout << "Time since last call: " << elapsed.count() << "\n";
        return elapsed.count();
    }

private:
    std::chrono::steady_clock::time_point last_time;
};

extern "C" NTSYSAPI NTSTATUS NTAPI NtSetTimerResolution(ULONG DesiredResolution, BOOLEAN SetResolution, PULONG CurrentResolution);

void SafeRelease(IUnknown* ppT) {
    if (ppT) {
       (ppT)->Release();
        ppT = NULL;
    }
}

void cleanup() {
    HANDLE hProcess = GetCurrentProcess();

    UINT exitCode = 0;

    // idk I tried to clean up right, we're uhhh gunna just scoot ya outta here bud. 
    TerminateProcess(hProcess, exitCode);

    if (hMouseHook) {
        UnhookWindowsHookEx(hMouseHook);
        hMouseHook = nullptr;
    }

    // Release COM and DirectX resources
    SafeRelease((IUnknown*&)adapter);
    SafeRelease((IUnknown*&)d3dfactory);
    SafeRelease((IUnknown*&)d2dFactory);
    SafeRelease((IUnknown*&)d2dDevice);
    SafeRelease((IUnknown*&)d2dContext);
    SafeRelease((IUnknown*&)dxgiDevice);
    SafeRelease((IUnknown*&)renderTargetView);
    SafeRelease((IUnknown*&)dcompDevice);
    SafeRelease((IUnknown*&)dcompTarget);
    SafeRelease((IUnknown*&)dcompVisual);
    SafeRelease((IUnknown*&)swapchain);
    SafeRelease((IUnknown*&)cudaTextureSwapchain);
    SafeRelease((IUnknown*&)cudaTexture);
    SafeRelease((IUnknown*&)cudaTextureBackBuffer);
    SafeRelease((IUnknown*&)cudaTextureUAV);
    SafeRelease((IUnknown*&)outputTextureSwapchain);
    SafeRelease((IUnknown*&)outputTexture);
    SafeRelease((IUnknown*&)outputTextureBackBuffer);
    SafeRelease((IUnknown*&)outputTextureUAV);
    SafeRelease((IUnknown*&)d3dDevice);
    SafeRelease((IUnknown*&)d3dContext);
    SafeRelease((IUnknown*&)dxgiBackBuffer);
    SafeRelease((IUnknown*&)d2dBitmapBackBuffer);
    SafeRelease((IUnknown*&)d2dTextureBackBuffer);
    SafeRelease((IUnknown*&)desktopTexture);
    SafeRelease((IUnknown*&)desktopShaderResourceView);
    SafeRelease((IUnknown*&)outputDuplication);
    SafeRelease((IUnknown*&)ScanComputeShader);
    SafeRelease((IUnknown*&)CopyComputeShader);
    SafeRelease((IUnknown*&)xBuffer);
    SafeRelease((IUnknown*&)yBuffer);
    SafeRelease((IUnknown*&)xUAV);
    SafeRelease((IUnknown*&)yUAV);
    SafeRelease((IUnknown*&)xReadback);
    SafeRelease((IUnknown*&)yReadback);
    SafeRelease((IUnknown*&)regionXBuffer);
    SafeRelease((IUnknown*&)regionYBuffer);
    SafeRelease((IUnknown*&)regionXUAV);
    SafeRelease((IUnknown*&)regionYUAV);
    SafeRelease((IUnknown*&)debugBuffer);
    SafeRelease((IUnknown*&)debugUAV);
    SafeRelease((IUnknown*&)samplerState);

    // Cleanup CUDA resources
    if (cudaResource) {
        cudaGraphicsUnregisterResource(cudaResource);
        cudaResource = nullptr;
    }
    if (cudaTextureArray) {
        cudaFreeArray(cudaTextureArray);
        cudaTextureArray = nullptr;
    }

    // Release memory allocated to rgbCudaArray
    if (rgbCudaArray) {
        delete[] rgbCudaArray;
        rgbCudaArray = nullptr;
    }
}