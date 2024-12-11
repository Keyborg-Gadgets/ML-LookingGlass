#include "StandAloneCapture.h"

/// 
/// 
///  All variables are defined in globals.h
///  
///  
///  

//static winrt::Windows::Graphics::Capture::GraphicsCaptureItem winrt_capture_create_item(IGraphicsCaptureItemInterop* interop_factory, HMONITOR monitor);
static void InitializeCapture();
HRESULT CaptureFrame();
Timer timer;

int main()
{
    // Get some global screen and system settings
    int titleBarHeight = GetSystemMetrics(SM_CYCAPTION);
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
    // So the dxgi surface is a double bound backbuffer. Context rules still apply. No concurrent access. But you can use D2D1 or 2D texture methods before the swap.
    // Draw a green circle on the screen. I use the rtv for clearing etc.
    // Render();
    //CreateRenderTargetView(d3dDevice, swapchain, &renderTargetView, &d2dTextureBackBuffer);
    //InitializeComputeShader();
    //CreateDesktopSRV();
    InitializeCapture();

    running = true;
    while(true){
        timer.sinceLast();
        CaptureFrame();

        /*CopyDirtyRects();*/
        /*swapchain->Present(1, 0);*/
        //std::vector<unsigned int> xy = ScanTexture();
        //if (xy.size() > 0) {
        //    xOfWindow = xy[0];
        //    yOfWindow = xy[1];
        //}
        std::cout << "x:" << xOfWindow << " " << "y:" << yOfWindow << "\n";

        Render();
        outputDuplication->ReleaseFrame();
    }

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

// This worked great across a number of implementations. I never used debug libraries/headers though, so 
// when I started this project, I started with them to see if it would make my life easier. Well cool, when I 
// compile for release, the on_frame_arrived portion hangs. I don't know if the H2 upgrade caused it or what.
// But it's running at 20ms/cycle on a 144hz screen. It never had that problem before, even in debug, and I can't 
// compile for release. I really dislike the winrt ecosystem. Sounds great but I can never debug or troubleshoot 
// it and StackOverflow is pretentious AF. Cool story bro, chatGPT can tell me what a pointer is. What are you 
// here for? I can't google "winrt screen cap slow" "release/debug compile hang" (so much noise) etc. You can't 
// have a conversation anymore. And the C++ space is so uninviting. I sit in front of this computer 16 hours a day 
// somedays. I can RTFM. Every once in a while, it would be nice to have help with the in-between. If I do any 
// more exercises left to the reader I will be the worlds strongest man.

//struct winrt_capture {
//    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
//    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool frame_pool{ nullptr };
//    winrt::Windows::Graphics::Capture::GraphicsCaptureSession session{ nullptr };
//    winrt::Windows::Graphics::Capture::GraphicsCaptureItem::Closed_revoker closed;
//    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::FrameArrived_revoker frame_arrived;
//
//    void on_closed(winrt::Windows::Graphics::Capture::GraphicsCaptureItem const&, winrt::Windows::Foundation::IInspectable const&) {
//        running = false;
//    }
//
//    void on_frame_arrived(winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& sender, winrt::Windows::Foundation::IInspectable const&) {
//        if (running) {
//            timer.sinceLast();
//            const auto frame = sender.TryGetNextFrame();
//            /*const auto frame_content_size = frame.ContentSize();
//            auto frame_surface = GetDXGIInterfaceFromObject<ID3D11Texture2D>(frame.Surface());
//
//            D3D11_TEXTURE2D_DESC desc;
//            frame_surface->GetDesc(&desc);*/
//
//            //auto position = ScanTexture(frame_surface.get());
//
//            //if (!position.empty()) {
//            //    uint32_t xOfWindow = position[0] - 10;
//            //    uint32_t yOfWindow = position[1] + 11;
//
//            //    D3D11_BOX srcBox = { xOfWindow, yOfWindow, 0, xOfWindow + imgsz - 1, yOfWindow + imgsz - 1, 1 };
//            //    d3dContext->CopySubresourceRegion(texture, 0, 0, 0, 0, frame_surface.get(), 0, &srcBox);
//            //}
//        }
//        else {
//            stop();
//        }
//    }
//
//    void stop() {
//        closed.revoke();
//        frame_arrived.revoke();
//
//        if (session) {
//            session.Close();
//            session = nullptr;
//        }
//
//        if (frame_pool) {
//            frame_pool.Close();
//            frame_pool = nullptr;
//        }
//
//        running = false;
//    }
//};
//
//struct winrt_capture* capture;
//
//static winrt::Windows::Graphics::Capture::GraphicsCaptureItem
//winrt_capture_create_item(IGraphicsCaptureItemInterop* const interop_factory,
//    HMONITOR monitor) {
//    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item = { nullptr };
//    assert(monitor);
//    try {
//        const HRESULT hr = interop_factory->CreateForMonitor(
//            monitor,
//            winrt::guid_of<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>(),
//            reinterpret_cast<void**>(winrt::put_abi(item)));
//        if (FAILED(hr))
//            printf("CreateForMonitor (0x%08X)", hr);
//    }
//    catch (winrt::hresult_error& err) {
//        printf("CreateForMonitor (0x%08X): %s", err.code().value,
//            winrt::to_string(err.message()).c_str());
//    }
//    catch (...) {
//        printf("CreateForMonitor (0x%08X)", winrt::to_hresult().value);
//    }
//    return item;
//}
//
//static void InitializeCapture() {
//    winrt::init_apartment(winrt::apartment_type::single_threaded);
//    capture = new winrt_capture{};
//    auto activation_factory = winrt::get_activation_factory<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>();
//    auto interop_factory = activation_factory.as<IGraphicsCaptureItemInterop>();
//    auto item = winrt_capture_create_item(interop_factory.get(), monitor);
//    if (!item) exit(1);
//
//    auto pixelFormat = winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized;
//    auto frame_pool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(direct3DDevice, pixelFormat, 1, item.Size());
//    auto session = frame_pool.CreateCaptureSession(item);
//    capture->item = item;
//    capture->frame_pool = frame_pool;
//    capture->session = session;
//    capture->closed = item.Closed(winrt::auto_revoke, { capture, &winrt_capture::on_closed });
//    capture->frame_arrived = frame_pool.FrameArrived(winrt::auto_revoke, { capture, &winrt_capture::on_frame_arrived });
//    session.IsCursorCaptureEnabled(false);
//    InitializeComputeShader();
//    session.StartCapture();
//}