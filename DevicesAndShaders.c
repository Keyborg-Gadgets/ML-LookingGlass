#define COBJMACROS
#include "DevicesAndShaders.h"

IDXGIAdapter1*GetDefaultAdapter() {
  IDXGIFactory1 *factory = NULL;
  HRESULT hr = CreateDXGIFactory1(&IID_IDXGIFactory1, (void **)&factory);
  assert(SUCCEEDED(hr));

  IDXGIAdapter1 *adapter = NULL;
  hr = IDXGIFactory1_EnumAdapters1(factory, 0, &adapter);
  assert(SUCCEEDED(hr));
  
  IDXGIFactory1_Release(factory);

  return adapter;
}

IDXGIDevice1* GetDxgiDevice(ID3D11Device1* device) {
    IDXGIDevice1* dxgi_dev = NULL;
    HRESULT hr = ID3D11Device_QueryInterface(device, &IID_IDXGIDevice1,
        (void**)&dxgi_dev);
    assert(SUCCEEDED(hr));
    return dxgi_dev;
}

IDXGISwapChain1 *CreateSwapChain(ID3D11Device1 *device, int width, int height) {
  HRESULT hr = 0;
  IDXGIDevice1 *dxgi_dev = NULL;
  hr = ID3D11Device_QueryInterface(device, &IID_IDXGIDevice1,
                                   (void **)&dxgi_dev);
  assert(SUCCEEDED(hr));

  IDXGIAdapter1 *adapter = NULL;
  hr = IDXGIDevice_GetParent(dxgi_dev, &IID_IDXGIAdapter1, (void **)&adapter);
  assert(SUCCEEDED(hr));

  IDXGIFactory2 *factory = NULL;
  hr = IDXGIAdapter1_GetParent(adapter, &IID_IDXGIFactory2, (void **)&factory);
  assert(SUCCEEDED(hr));

  DXGI_SWAP_CHAIN_DESC1 swapchain_desc = {
      .Width = width,
      .Height = height,
      .Format = DXGI_FORMAT_B8G8R8A8_UNORM,
      .SampleDesc = {.Count = 1},
      .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
      .BufferCount = 2,
      .Scaling = DXGI_SCALING_STRETCH,
      .SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
      .AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED,
  };

  IDXGISwapChain1 *swapchain = NULL;
  hr = IDXGIFactory2_CreateSwapChainForComposition(factory, (IUnknown *)device,
                                            &swapchain_desc, NULL,
                                            &swapchain);
  assert(SUCCEEDED(hr));

  return swapchain;
}

void GenerateRandomString(char* str, size_t length) {
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (size_t i = 0; i < length - 1; i++) {
        str[i] = charset[rand() % (sizeof(charset) - 1)];
    }
    str[length - 1] = '\0';
}

BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
    MONITORINFO monitorInfo;
    monitorInfo.cbSize = sizeof(MONITORINFO);

    if (GetMonitorInfo(hMonitor, &monitorInfo)) {
        RECT rect = monitorInfo.rcMonitor;
        int width = rect.right - rect.left;
        int height = rect.bottom - rect.top;

        // Store width and height in the pointer passed via dwData
        int* dimensions = (int*)dwData;
        dimensions[0] = width;
        dimensions[1] = height;

        // Stop enumerating after the first monitor
        return FALSE;
    }
    return TRUE; // Continue enumeration
}

void GetFirstMonitorSize(int* width, int* height) {
    int dimensions[2] = { 0, 0 };
    EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, (LPARAM)dimensions);
    *width = dimensions[0];
    *height = dimensions[1];
}

IDXGISwapChain1* CreateSwapChainForUAV(ID3D11Device1* device, int width, int height) {
    char className[16];
    GenerateRandomString(className, sizeof(className));
    HINSTANCE hInstance = GetModuleHandle(NULL);
    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = DefWindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = className;

    RegisterClass(&wc);

    HWND hwndDummy = CreateWindowEx(
        WS_EX_TOOLWINDOW | WS_EX_LAYERED,                   
        className,         
        className,      
        WS_OVERLAPPEDWINDOW, 
        width, height, width, height,         
        NULL,               
        NULL,             
        hInstance,           
        NULL               
    );

    ShowWindow(hwndDummy, SW_SHOWMINNOACTIVE);

    HRESULT hr = 0;
    IDXGIDevice1* dxgi_dev = NULL;
    hr = ID3D11Device_QueryInterface(device, &IID_IDXGIDevice1,
        (void**)&dxgi_dev);
    assert(SUCCEEDED(hr));

    IDXGIAdapter1* adapter = NULL;
    hr = IDXGIDevice_GetParent(dxgi_dev, &IID_IDXGIAdapter1, (void**)&adapter);
    assert(SUCCEEDED(hr));

    IDXGIFactory2* factory = NULL;
    hr = IDXGIAdapter1_GetParent(adapter, &IID_IDXGIFactory2, (void**)&factory);
    assert(SUCCEEDED(hr));

    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {
    .Width = width,
    .Height = height,
    .Format = DXGI_FORMAT_B8G8R8A8_UNORM,
    .SampleDesc = {.Count = 1 },
    .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_UNORDERED_ACCESS,
    .BufferCount = 3,
    .Scaling = DXGI_SCALING_NONE,
    .Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING,
    .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
    .AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED
    };

    IDXGISwapChain1* swapchain = NULL;
    hr = IDXGIFactory2_CreateSwapChainForHwnd(
        factory,
        (IUnknown*)device,
        hwndDummy,              
        &swapChainDesc,
        NULL,                   
        NULL,                   
        &swapchain
    );
    assert(SUCCEEDED(hr));

    return swapchain;
}

void FinalSwap(IDXGISwapChain1 *swapchain) {
  IDXGISwapChain1_Present(swapchain, 1, 0);
}

void Cleanup(ID3D11DeviceContext1 **ctx, ID3D11Device1 **device,
             IDXGISwapChain1 **swapchain, ID3D11RenderTargetView **rtv,
             ID3D11Texture2D **texture, ID3D11Texture2D **backBuffer) {
  if (*rtv) {
    ID3D11RenderTargetView_Release(*rtv);
    *rtv = NULL;
  }
  if (*ctx) {
    ID3D11DeviceContext1_Release(*ctx);
    *ctx = NULL;
  }
  if (*device) {
    ID3D11Device1_Release(*device);
    *device = NULL;
  }
  if (*swapchain) {
    IDXGISwapChain1_Release(*swapchain);
    *swapchain = NULL;
  }
  if (*texture) {
    ID3D11Texture2D_Release(*texture);
    *texture = NULL;
  }
  if (*backBuffer) {
    ID3D11Texture2D_Release(*backBuffer);
    *backBuffer = NULL;
  }
}

void CreateRenderTargetView(ID3D11Device1 *device, IDXGISwapChain1 *swapchain,
                            ID3D11RenderTargetView **rtv,
                            ID3D11Texture2D **backBuffer) {
  HRESULT hr = 0;
  hr = IDXGISwapChain1_GetBuffer(swapchain, 0, &IID_ID3D11Texture2D,
                                 (void **)backBuffer);
  assert(SUCCEEDED(hr));
  hr = ID3D11Device_CreateRenderTargetView(
      device, (ID3D11Resource *)*backBuffer, NULL, rtv);
  assert(SUCCEEDED(hr));
}

void CreateDeviceAndContext(IDXGIAdapter1* adapter, ID3D11Device1** device,
    ID3D11DeviceContext1** ctx, IDXGIFactory2** d3dfactory) {
    HRESULT hr = 0;
    D3D_FEATURE_LEVEL feature_levels[] = { D3D_FEATURE_LEVEL_11_1 };
    ID3D11Device* base_device = NULL;
    ID3D11DeviceContext* base_ctx = NULL;

    hr = D3D11CreateDevice(
        (IDXGIAdapter*)adapter, 
        D3D_DRIVER_TYPE_UNKNOWN, 
        NULL, D3D11_CREATE_DEVICE_BGRA_SUPPORT /*| D3D11_CREATE_DEVICE_DEBUG*/, feature_levels,
        ARRAYSIZE(feature_levels), D3D11_SDK_VERSION, &base_device, NULL,
        &base_ctx);
    assert(SUCCEEDED(hr));

    hr = ID3D11Device_QueryInterface(base_device, &IID_ID3D11Device1,
        (void**)device);
    assert(SUCCEEDED(hr));

    hr = ID3D11DeviceContext_QueryInterface(base_ctx, &IID_ID3D11DeviceContext1,
        (void**)ctx);
    assert(SUCCEEDED(hr));

    IDXGIDevice1* dxgi_dev = NULL;
    hr = ID3D11Device_QueryInterface(base_device, &IID_IDXGIDevice1,
        (void**)&dxgi_dev);
    assert(SUCCEEDED(hr));

    hr = IDXGIDevice_GetParent(dxgi_dev, &IID_IDXGIAdapter1, (void**)&adapter);
    assert(SUCCEEDED(hr));

    hr = IDXGIAdapter1_GetParent(adapter, &IID_IDXGIFactory2, (void**)&d3dfactory);
    assert(SUCCEEDED(hr));

    ID3D11Device_Release(base_device);
    ID3D11DeviceContext_Release(base_ctx);
}






















































