#pragma once
#include "pch.h"

std::string exeDir;
HRESULT hr = S_OK;
uint32_t imgsz = 1024;
uint32_t enginesz = 640;
uint32_t xOfWindow = 0;
uint32_t yOfWindow = 0;
uint32_t xOfMouse = 0;
uint32_t yOfMouse = 0;
HHOOK hMouseHook = nullptr;

bool running = false;
HWND overlayHwnd;
HWND lookingGlassHwnd;
HMONITOR monitor;
HINSTANCE hInstance;
std::atomic<bool> texture_written = false;
int titleBarHeight;

unsigned int monitor_width = 0;
unsigned int monitor_height = 0;

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
// 
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

ID3D11Buffer* xyOutputBuffer = nullptr;
ID3D11UnorderedAccessView* xyOutputBufferUAV = nullptr;
ID3D11Buffer* xyOutputBufferReadback = nullptr;
ID3D11Buffer* controlOutputBuffer = nullptr;
ID3D11UnorderedAccessView* controlOutputBufferUAV = nullptr;
ID3D11Buffer* controlOutputBufferReadback = nullptr;

ID3D11Buffer* regionBuffer = nullptr;
ID3D11UnorderedAccessView* regionBufferUAV = nullptr;

ID3D11Buffer* xBuffer = nullptr;
ID3D11Buffer* yBuffer = nullptr;
ID3D11UnorderedAccessView* xUAV = nullptr;
ID3D11UnorderedAccessView* yUAV = nullptr;

struct xyStruct {
    uint32_t x;
    uint32_t y;
};

struct controlStruct {
    uint32_t triggerScan;
    uint32_t scanComplete;
};

struct regionStruct {
    uint32_t width;
    uint32_t height;
};

ID3D11SamplerState* samplerState = nullptr;

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


ID3D11RenderTargetView* rtv;