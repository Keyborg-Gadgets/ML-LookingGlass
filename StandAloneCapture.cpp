#include "StandAloneCapture.h"
Timer timer;
int main()
{
    // Set up some global info
    hInstance = GetModuleHandle(NULL);
    monitor = MonitorFromPoint({ 0, 0 }, MONITOR_DEFAULTTOPRIMARY);
    GetScreenSizeFromHMonitor(monitor, monitor_width, monitor_height);
    
    // I don't like when the app pauses if you click things. It's called the modal loop being paused. Threading stuff avoids it. 
    std::thread([]() { CreateOverlayAndLookingGlass(); }).detach();
    
    // Adapters and the sort. This uses the modern dxgi framework. Why is some of it C? C (see, har har har) the faq.
    adapter = GetDefaultAdapter();
    CreateDeviceAndContext(adapter, &d3dDevice, &d3dContext, &d3dfactory);
    dxgiDevice = GetDxgiDevice(d3dDevice);
    swapchain = CreateSwapChain(d3dDevice, monitor_width, monitor_height);
    
    // Prep an empty texture for the dxgi capture
    Create2DTexture(&desktopTexture, monitor_width, monitor_height);
    
    // This gives you 2 vars bound to the same dxgi surface allowing you to use either d2d or 2d texture methods.
    CreateD2DDevice();
    CreateDCompDevice();
    InitializeComputeShader();

    // This is the render target view for the d2dTextureBackBuffer. I use it to clear the texture. 
    CreateRenderTargetView(d3dDevice, swapchain, &renderTargetView, &d2dTextureBackBuffer);

    InitializeCapture();

    running = true;

    SetLayeredWindowAttributes(lookingGlassHwnd, 0, 255, LWA_ALPHA);
    ShowWindow(lookingGlassHwnd, SW_SHOWDEFAULT);
    std::thread([]() { 
        while (true) {
            // timer.sinceLast();
            CaptureFrame();
            // So in the early days of trying to get this just right, I didn't know where my delay was coming from
            // Is it the getclient functions? Was it updating window properties etc? So I built a compute shader to
            // Find an icon in the capture texture. It's hella fast and it is guarenteed to be insync with the frame. 
            // So I left it because it's fun and clever. This is a playground demo too. 
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
                // std::cout << "x:" << xOfWindow << " " << "y:" << yOfWindow << "\n";
            }
            outputDuplication->ReleaseFrame();
        }
    }).detach();

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
    };

	return 0;
}


