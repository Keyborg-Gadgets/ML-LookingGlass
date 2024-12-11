#pragma once
#include "pch.h"

HRESULT hr = S_OK;
int imgsz = 1024;
int xOfWindow = 0;
int yOfWindow = 0;
bool running = false;
HWND overlayHwnd;
HWND lookingGlassHwnd;
HMONITOR monitor;
HINSTANCE hInstance;
std::atomic<bool> texture_written = false;
D3D11_BOX srcBox;
int titleBarHeight;

int monitor_width = 0;
int monitor_height = 0;

//winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice direct3DDevice;
IDXGIAdapter1* adapter = nullptr;
//winrt::com_ptr<IDXGIDevice> dxgiDevice; 

IDXGIFactory2* d3dfactory = nullptr;
ID2D1Factory2* d2dFactory = nullptr;
ID2D1Device1* d2dDevice = nullptr;
ID2D1DeviceContext* d2dContext = nullptr;
IDXGISurface2* dxgiBackBuffer = nullptr;
ID2D1Bitmap1* d2dBitmapBackBuffer = nullptr;
ID3D11Texture2D* d2dTextureBackBuffer = nullptr;
IDXGIDevice1* dxgiDevice = nullptr;
ID3D11RenderTargetView* renderTargetView = nullptr;

IDCompositionDevice* dcompDevice = nullptr;
IDCompositionTarget* dcompTarget = nullptr;
IDCompositionVisual* dcompVisual = nullptr;
IDXGISwapChain1* swapchain = nullptr;
ID3D11Device1* d3dDevice = nullptr;
ID3D11DeviceContext1* d3dContext = nullptr;
ID3D11Texture2D* desktopTexture = nullptr;
ID3D11ShaderResourceView* desktopShaderResourceView = nullptr;
IDXGIOutputDuplication* outputDuplication = nullptr;

ID3D11ComputeShader* computeShader = nullptr;
ID3D11Buffer* outputBuffer = nullptr;
ID3D11UnorderedAccessView* outputBufferUAV = nullptr;
ID3D11Buffer* outputBufferReadback = nullptr;
ID3D11SamplerState* samplerState = nullptr;

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