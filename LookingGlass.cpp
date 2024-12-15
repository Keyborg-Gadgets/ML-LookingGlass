#include "LookingGlass.h"
Timer timer;
int main()
{
    // Gotta go fast.
    NtSetTimerResolution(5000, TRUE, &currentRes);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // Set up some global info used in a ton of stuff.
    exeDir = getExecutableDirectory();
    hInstance = GetModuleHandle(NULL);
    monitor = MonitorFromPoint({ 0, 0 }, MONITOR_DEFAULTTOPRIMARY);
    GetScreenSizeFromHMonitor(monitor, monitor_width, monitor_height);

    // Create window, cool stuff in here.
    std::thread([]() { CreateOverlayAndLookingGlass(); }).detach();
    
    // Adapters and the sort. This uses the modern dxgi framework. Why is some of it C? C (see, har har har) the faq.
    // Create a DXGI adapter. The is the foundation for global directx operations. If you play with this code just be 
    // aware that there's tiny nuances. Shit just wont work and you won't know whats going on. If you get into shared 
    // textures and mutexes for example the sharing flags are different vs traditional. Nothing is thread safe in terms
    // of the device context currently. All my operations are primed and operate in the same thread. You can look into 
    // enter graphics if you need true threading. I don't use it here but this has the d3d11_4 header see:ID3D11Multithread 
    adapter = GetDefaultAdapter();
    CreateDeviceAndContext(adapter, &d3dDevice, &d3dContext, &d3dfactory);
    dxgiDevice = GetDxgiDevice(d3dDevice);
    swapchain = CreateSwapChain(d3dDevice, monitor_width, monitor_height);

    // So this is goofy, but the whole dxgi/d3d space is a cluster fuck and sometimes things won't work and you don't know why, debug enabled
    // doesn't help and I think fundementally it's because half the examples on the internet are 20 years old. I cannot for the life of me
    // create a UAV that can be written to, without making it with a swap chain. Oh I can create it. The shader can compile and run. No debug 
    // output. It just doesn't work silently. I had to guess this shit. I thought to my self hey, I have this other thing that works, it was
    // made by this thing, so I make swap chains to dummy hwnds because it friggin works. Then I get the texture from the swap chain and it 
    // friggin works. And it's bulky but fast as fuck boi. Just like me. 
    outputTextureSwapchain = CreateSwapChainForUAV(d3dDevice, monitor_width, monitor_height);
    cudaTextureSwapchain = CreateSwapChainForUAV(d3dDevice, enginesz, enginesz);
    hr = outputTextureSwapchain->GetBuffer(0, IID_PPV_ARGS(&outputTexture));
    hr = cudaTextureSwapchain->GetBuffer(0, IID_PPV_ARGS(&cudaTexture));

    CreateD2DDevice();
    CreateDCompDevice();

    Create2DTexture(&desktopTexture, monitor_width, monitor_height);
    InitializeComputeShader();

    // Make sure our window is up
    while (!winDone) {};
    // Start the ixdgi capture session
    InitializeCapture();
    running = true;

    HWND hwnd = GetConsoleWindow();
    ShowWindow(hwnd, SW_HIDE);
#ifdef _DEBUG
    ShowWindow(hwnd, SW_SHOW);
#endif
    std::thread([]() { 
        while (true) {
#ifdef _DEBUG
            timer.sinceLast();
#endif
            CaptureFrame();

            if (xOfWindow == -1 || yOfWindow == -1) {
                int x = (monitor_width - imgsz) / 2;
                int y = (monitor_height - imgsz) / 2;
                SetWindowPos(lookingGlassHwnd, HWND_TOPMOST, x, y, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
            }
            // ScanTexture Evolved a lot. Originally I didn't know where the window jitter was coming from, I thought maybe it was 
            // an issue with calculating the screen space, so I wanted a way to know I was in sync with the frame. I create a special
            // ICON for the looking glass window and I scan for it using a compute shader. I figured I might as well copy and interpolate
            // there as well. It ended up working really well. I could use a single structured buffer but something was weird and I decided
            // not to play with it and just use a single uint. 
            ScanTexture(desktopTexture);
            d3dContext->CopyResource(d2dTextureBackBuffer, outputTexture);
#ifdef _DEBUG
            // This shows your interpolated texture
            d3dContext->CopySubresourceRegion(d2dTextureBackBuffer, 0, 0, 0, 0, cudaTexture,0, nullptr);
#endif
            if (xOfWindow != -1 && yOfWindow != -1) {
                D3D11_BOX srcBox;
                srcBox.left = xOfWindow;
                srcBox.top = yOfWindow;
                srcBox.front = 0;
                srcBox.right = xOfWindow + imgsz - 1;
                srcBox.bottom = yOfWindow + imgsz - 1;
                srcBox.back = 1;

                swapchain->Present(1, 0);
#ifdef _DEBUG
                std::cout << "x:" << xOfWindow << " " << "y:" << yOfWindow << "\n";
#endif 
            }
            outputDuplication->ReleaseFrame();
        }
    }).detach();

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
    };

	return 0;
}