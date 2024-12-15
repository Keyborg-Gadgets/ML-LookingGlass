#include "LookingGlass.h"
Timer timer;
int main()
{
    NtSetTimerResolution(5000, TRUE, &currentRes);
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) { 
        std::cerr << "Failed to set process priority. Error Code: " << GetLastError() << std::endl; 
    }
    else { std::cout << "Process priority set to high." << std::endl; 
    }

    // Set up some global info
    exeDir = getExecutableDirectory();
    hInstance = GetModuleHandle(NULL);
    monitor = MonitorFromPoint({ 0, 0 }, MONITOR_DEFAULTTOPRIMARY);
    // Set the monitor width and height. Used for the overlay and globally for a ton of stuff
    GetScreenSizeFromHMonitor(monitor, monitor_width, monitor_height);
    
    // I don't like when the app pauses if you click things. It's called the modal loop being paused. Threading stuff avoids it. 
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
    // doesn't help and I think funcdementally it's because half the examples on the internet are 20 years old. I cannot for the life of me
    // create a UAV that can be written to, without making it with a swap chain. Oh I can create it. The shader can compile and run. No debug 
    // output. It just doesn't work silently. I had to guess this shit. I thought to my self hey, I have this other thing that works, it was
    // made by this thing, so I make swap chains to dummy hwnds because it friggin works. Then I get the texture from the swap chain and it 
    // friggin works. And it's bulky but fast as fuck boi. Just like me. 
    outputTextureSwapchain = CreateSwapChainForUAV(d3dDevice, monitor_width, monitor_height);
    cudaTextureSwapchain = CreateSwapChainForUAV(d3dDevice, enginesz, enginesz);
    hr = outputTextureSwapchain->GetBuffer(0, IID_PPV_ARGS(&outputTexture));
    assert(SUCCEEDED(hr));
    hr = cudaTextureSwapchain->GetBuffer(0, IID_PPV_ARGS(&cudaTexture));
    assert(SUCCEEDED(hr) && cudaTexture != nullptr);

    // This gives you 2 vars bound to the same dxgi surface allowing you to use either d2d or 2d texture methods. 
    CreateD2DDevice();

    // I'm going to do the frame drawings for the detections with d2d which is not the same as 2d texture. d2d is like 
    // part of the composition engine. I'm not really familiar with it. The TLDR is that, thats how windows displays things
    // for a very long time now. I cover the jitter in the docs and why directly accelerated rendering didn't work. Anywho,
    // since this is a play ground, I'm doing it here. Normally I'd just do it in cuda and call it a day.
    CreateDCompDevice();

    //hr = cudaTextureSwapchain->QueryInterface(__uuidof(ID3D11Texture2D), (void**)(&cudaTexture));
    //assert(SUCCEEDED(hr) && outputTexture != nullptr);
    //if (SUCCEEDED(hr)) {
    //    D3D11_TEXTURE2D_DESC desc;
    //    cudaTexture->GetDesc(&desc);
    //    printf("Back buffer format: %d, width: %d, height: %d\n", desc.Format, desc.Width, desc.Height);
    //}
    //hr = outputTextureBackBuffer->QueryInterface(__uuidof(ID3D11Texture2D), (void**)(&outputTexture));
    //assert(SUCCEEDED(hr) && outputTexture != nullptr);

    Create2DTexture(&desktopTexture, monitor_width, monitor_height);
    //Create2DTexture(&outputTexture, monitor_width, monitor_height, true);
    //Create2DTexture(&cudaTexture, enginesz, enginesz, true);

    // I'm not going to get too into shaders. They're just cuda kernels but windows flavored. Threads and all that. 
    // Scanning shader for tracking the window, used by ScanTexture. Idk. I'm not going to get into shaders. They're not hard 
    // once you get it, and chatgpt can get you 90% of the way there, but I mostly prefer cuda. They're cuda kernels essentailly
    // look at it as a cu file. It launches threads and all that. I could probably leave this shader open and have some sort of
    // abilities to call functions in it. I could just have it spin, and perform tasks based on a buffer or something. There's 
    // fun stuff here but I got other stuff to work on. I bet you could manipulate everything here SUPER fast though.
    InitializeComputeShader();

    // This is the render target view for the d2dTextureBackBuffer. I use it to clear the texture. 
    CreateRenderTargetView(d3dDevice, swapchain, &renderTargetView, &d2dTextureBackBuffer);
    while (!winDone) {};
    // Start the ixdgi capture session
    InitializeCapture();

    running = true;

    // I use mouse hooks and some other stuff to manage window transparency. I launch the window in a thread though before requisit components are ready, so I just wait to show so the window is
    // immedaitly respondent when you see it.


    //CreateRenderTargetView(d3dDevice, swapchain, &rtv, &outputTexture);

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

            // So in the early days of trying to get this just right, I didn't know where my delay was coming from
            // Is it the getclient functions? Was it updating window properties etc? So I built a compute shader to
            // Find an icon in the capture texture. It's hella fast and it is guarenteed to be insync with the frame. 
            // So I left it because it's fun and clever but the window proc probably handles it just fine. This is a playground demo too. 
            
            //const float clearColor[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
            //d3dContext->ClearRenderTargetView(rtv, clearColor);

            ScanTexture(desktopTexture);
            d3dContext->CopyResource(d2dTextureBackBuffer, outputTexture);
#ifdef _DEBUG
            d3dContext->CopySubresourceRegion(
                d2dTextureBackBuffer, 0, 0, 0, 0, cudaTexture,0, nullptr);
#endif

            if (xOfWindow != -1 && yOfWindow != -1) {
                D3D11_BOX srcBox;
                srcBox.left = xOfWindow;
                srcBox.top = yOfWindow;
                srcBox.front = 0;
                srcBox.right = xOfWindow + imgsz - 1;
                srcBox.bottom = yOfWindow + imgsz - 1;
                srcBox.back = 1;
                // You can swap the clear color here.The last value is alpha, change it to see the mask.

                // These are 2d texture functions, seprate from the bitmap stuff, but share the same surface
                //d3dContext->CopySubresourceRegion(d2dTextureBackBuffer, 0, static_cast<uint32_t>(xOfWindow),
                //    static_cast<uint32_t>(yOfWindow), 0, desktopTexture, 0, &srcBox);

                
        

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


