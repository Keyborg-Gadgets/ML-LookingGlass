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

    // Create window, cool stuff in here. Idk what blocks what doesn't. I don't think you can escape the modal loop
    // but the window and message pumps get their own thread.
    std::thread([]() { CreateOverlayAndLookingGlass(); }).detach();
    
    // Adapters and the sort. This uses the modern dxgi framework, the foundation for global directx operations. If you 
    // play with this code just be aware that there's tiny nuances. Shit just wont work and you won't know whats going on. 
    // If you get into shared textures and mutexes for example the sharing flags are different vs traditional. Nothing is 
    // thread safe in terms of the device context currently. All my operations are primed and operate in the same thread. 
    // You can look into enter graphics if you need true threading. I don't use it, but this has d3d11_4.h see:ID3D11Multithread
    // I think they're called resource bariers. I'll talk about it in DevicesAndShaderscpp.h, but even within a thread you must
    // cause the context to flush the resource
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
    InitRTdetr();
    // Show/Hide the console window for debug vs release
    HWND chwnd = GetConsoleWindow();
    ShowWindow(chwnd, SW_HIDE);
#ifdef _DEBUG
    ShowWindow(chwnd, SW_SHOW);
#endif
    // Start the ixdgi capture session
    InitializeCapture();

    // There's a commit in there that has a ton of good onnx stuff. The model was probably broken at the
    // the time but if I gotta get in the bones to figure that out we're just stayin in the bones.
    // Onnx could be cool. It feels cool. idk, i cant do any more heavy abstractions. I'm fried on it.
    /*InitializeOnnx();*/

    std::thread([]() { 
        while (running) {
#ifdef _DEBUG
            timer.sinceLast();
#endif
            // Request a dxgi frame
            CaptureFrame();
            // There's very few sceneraios you can get something infront of the window. If you do, we put it on top.
            if (xOfWindow == -1 || yOfWindow == -1) {
                int x = (monitor_width - imgsz) / 2;
                int y = (monitor_height - imgsz) / 2;
                SetWindowPos(lookingGlassHwnd, HWND_TOPMOST, x, y, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
            }
            // ScanTexture Evolved a lot. Originally I didn't know where the window jitter was coming from, I thought maybe it was 
            // an issue with calculating the screen space, so I wanted a way to know I was in sync with the frame. I create a special
            // ICON for the looking glass window and I scan for it using a compute shader. I figured I might as well copy and interpolate
            // there as well. It ended up working really well. I could use a single structured buffer but something was weird and I decided
            // not to play with it and just use a single uint x times. Cool stuff in here.
            ScanTexture(desktopTexture);
            outputDuplication->ReleaseFrame();
            if (xOfWindow == -1 || yOfWindow == -1) {
                continue;
            }
            // outputTexture at this point is a texture the size of the screen with an alpha channel that is transparent
            // around the imgsz and the imgsz region it's self is a reflection of the desktop. You also have an interpolated
            // cuda array that is sized to be used with the model.
            d3dContext->CopyResource(d2dTextureBackBuffer, outputTexture);
#ifdef _DEBUG
            // This shows your interpolated texture
            d3dContext->CopySubresourceRegion(d2dTextureBackBuffer, 0, 0, 0, 0, cudaTexture,0, nullptr);
#endif
            Detect();
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
        }
    }).detach();

    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    };

	return 0;
}