#include "StandAloneCapture.h"

/// 
/// 
///  All variables are defined in globals.h
///  
///  
///  

static void InitializeCapture();
HRESULT CaptureFrame();
Timer timer;

int main()
{
    // Get some global screen and system settings
    hInstance = GetModuleHandle(NULL);
    monitor = MonitorFromPoint({ 0, 0 }, MONITOR_DEFAULTTOPRIMARY);
    GetScreenSizeFromHMonitor(monitor, monitor_width, monitor_height);
    // Create the overlay in a thread, this give us the overlay hwnd
    std::thread([]() { CreateOverlayAndLookingGlass(); }).detach();
    /// set up the dxgi components
    adapter = GetDefaultAdapter();
    CreateDeviceAndContext(adapter, &d3dDevice, &d3dContext, &d3dfactory);
    dxgiDevice = GetDxgiDevice(d3dDevice);
    swapchain = CreateSwapChain(d3dDevice, monitor_width, monitor_height);
    // The dxgi capture output buffer
    Create2DTexture(&desktopTexture, monitor_width, monitor_height);
    // This will create 2 buffers that point to the same idxgi surface. One is a bitmap you can use with direct comp, the other you can use with 2d texture functions (copy etc)
    // I dont see anyone else doing this, but it seems to work.
    CreateD2DDevice();
    CreateDCompDevice();
    InitializeComputeShader();
    CreateRenderTargetView(d3dDevice, swapchain, &renderTargetView, &d2dTextureBackBuffer);
    InitializeCapture();

    running = true;
    // There's a thing called a modal loop. If you click on the cmd, it would pause the whole app. So I thread 
    // everything cause I don't like that.
    std::thread([]() { 
        while (true) {
            timer.sinceLast();
            CaptureFrame();
            ScanTexture(desktopTexture);
            if (xOfWindow != -1 && yOfWindow != -1) {
                D3D11_BOX srcBox;
                srcBox.left = xOfWindow;
                srcBox.top = yOfWindow;
                srcBox.front = 0;
                srcBox.right = xOfWindow + imgsz - 1;
                srcBox.bottom = yOfWindow + imgsz - 1;
                srcBox.back = 1;
                const float clearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                d3dContext->ClearRenderTargetView(renderTargetView, clearColor);
                d3dContext->CopySubresourceRegion(d2dTextureBackBuffer, 0, static_cast<uint32_t>(xOfWindow),
                    static_cast<uint32_t>(yOfWindow), 0, desktopTexture, 0, &srcBox);
                swapchain->Present(1, 0);
                std::cout << "x:" << xOfWindow << " " << "y:" << yOfWindow << "\n";
            }
            /*Render();*/
            outputDuplication->ReleaseFrame();
        }
    }).detach();

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
    };

	return 0;
}

static void InitializeCapture()
{
    IDXGIOutput* dxgiOutput = nullptr;
    HRESULT hr = adapter->EnumOutputs(0, &dxgiOutput);
    if (FAILED(hr) || !dxgiOutput)
    {
        exit(1);
    }

    IDXGIOutput1* dxgiOutput1 = nullptr;
    hr = dxgiOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&dxgiOutput1);
    dxgiOutput->Release();
    dxgiOutput = nullptr;
    if (FAILED(hr) || !dxgiOutput1)
    {
        exit(1);
    }

    hr = dxgiOutput1->DuplicateOutput(d3dDevice, &outputDuplication);
    dxgiOutput1->Release();
    dxgiOutput1 = nullptr;
}

HRESULT CaptureFrame()
{
    DXGI_OUTDUPL_FRAME_INFO frameInfo = {};
    IDXGIResource* desktopResource = nullptr;

    HRESULT hr = outputDuplication->AcquireNextFrame(
        1000, 
        &frameInfo,
        &desktopResource
    );

    if (hr == DXGI_ERROR_WAIT_TIMEOUT)
    {
        return S_FALSE;
    }

    if (FAILED(hr))
    {
        return hr;
    }

    hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&desktopTexture);
    desktopResource->Release();
    desktopResource = nullptr;
    if (FAILED(hr))
    {
        outputDuplication->ReleaseFrame();
        return hr;
    }
}