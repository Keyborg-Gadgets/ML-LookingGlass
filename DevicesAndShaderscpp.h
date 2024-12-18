#pragma once
#include "globals.h"
#include "DevicesAndShaders.h"

inline void CreateD2DDevice() {
    HRESULT hr;
    D2D1_FACTORY_OPTIONS options;
    memset(&options, 0, sizeof(options));

    hr = D2D1CreateFactory(
        D2D1_FACTORY_TYPE_SINGLE_THREADED,
        __uuidof(ID2D1Factory2),
        &options,
        reinterpret_cast<void**>(&d2dFactory)
    );
    assert(SUCCEEDED(hr) && "Failed to create D2D1 factory");

    hr = d2dFactory->CreateDevice(dxgiDevice, &d2dDevice);
    assert(SUCCEEDED(hr) && "Failed to create D2D1 device");

    hr = d2dDevice->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &d2dContext);
    assert(SUCCEEDED(hr) && "Failed to create D2D1 device context");

    hr = swapchain->GetBuffer(0, IID_PPV_ARGS(&dxgiBackBuffer));
    assert(SUCCEEDED(hr) && "Failed to get back buffer from swap chain");

    D2D1_BITMAP_PROPERTIES1 bitmapProperties;
    memset(&bitmapProperties, 0, sizeof(bitmapProperties));
    bitmapProperties.pixelFormat.format = DXGI_FORMAT_B8G8R8A8_UNORM;
    bitmapProperties.pixelFormat.alphaMode = D2D1_ALPHA_MODE_PREMULTIPLIED;
    bitmapProperties.bitmapOptions = D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW;

    hr = d2dContext->CreateBitmapFromDxgiSurface(dxgiBackBuffer, &bitmapProperties, &d2dBitmapBackBuffer);
    assert(SUCCEEDED(hr) && "Failed to create D2D1 bitmap from DXGI surface");
    hr = dxgiBackBuffer->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&d2dTextureBackBuffer));
    assert(SUCCEEDED(hr) && d2dTextureBackBuffer != nullptr);
    d2dContext->SetTarget(d2dBitmapBackBuffer);
}

inline void CreateDCompDevice() {
    HRESULT hr = DCompositionCreateDevice(dxgiDevice, __uuidof(IDCompositionDevice), reinterpret_cast<void**>(&dcompDevice));
    assert(SUCCEEDED(hr) && "Failed to create DComposition device");
    hr = dcompDevice->CreateTargetForHwnd(overlayHwnd, TRUE, &dcompTarget);
    assert(SUCCEEDED(hr) && "Failed to create DComposition target for HWND");
    hr = dcompDevice->CreateVisual(&dcompVisual);
    assert(SUCCEEDED(hr) && "Failed to create DComposition visual");
    hr = dcompVisual->SetContent(swapchain);
    assert(SUCCEEDED(hr) && "Failed to set content for DComposition visual");
    hr = dcompTarget->SetRoot(dcompVisual);
    assert(SUCCEEDED(hr) && "Failed to set root for DComposition target");
    hr = dcompDevice->Commit();
    assert(SUCCEEDED(hr) && "Failed to commit DComposition device");
}

inline void Render() {
    d2dContext->BeginDraw();

    d2dContext->Clear(D2D1::ColorF(D2D1::ColorF::Black, 0.0f));

    ID2D1SolidColorBrush* rectBrush = nullptr;
    D2D1_COLOR_F rectColor = D2D1::ColorF(D2D1::ColorF::LightBlue, 1.0f);
    hr = d2dContext->CreateSolidColorBrush(rectColor, &rectBrush);
    assert(SUCCEEDED(hr) && "Failed to create solid color brush");

    D2D1_RECT_F rectangle = D2D1::RectF(xOfWindow, yOfWindow, xOfWindow+imgsz-1, yOfWindow-titleBarHeight);
    d2dContext->FillRectangle(&rectangle, rectBrush);
    rectBrush->Release(); 

    hr = d2dContext->EndDraw();
    assert(SUCCEEDED(hr) && "Failed to end D2D drawing");

    hr = swapchain->Present(1, 0);
    assert(SUCCEEDED(hr) && "Failed to present swap chain");

    hr = dcompDevice->Commit();
    assert(SUCCEEDED(hr) && "Failed to commit DComposition device");
}

inline void Create2DTexture(ID3D11Texture2D** texture, int width, int height, bool unordered = false) {
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    if (unordered)
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    HRESULT hr = d3dDevice->CreateTexture2D(&desc, nullptr, texture);
}

HRESULT CreateUAVFromTexture(ID3D11Device* device, ID3D11Texture2D* texture, ID3D11UnorderedAccessView** uav) {
    if (!device || !texture || !uav) return E_INVALIDARG;

    D3D11_TEXTURE2D_DESC texDesc;
    texture->GetDesc(&texDesc);
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = texDesc.Format;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Texture2D.MipSlice = 0;
    HRESULT hr = device->CreateUnorderedAccessView(texture, &uavDesc, uav);
    if (FAILED(hr)) {
        std::cerr << "Failed to create UAV. Error Code: " << hr << std::endl;
    }

    return hr;
}

HRESULT CreateSRVFromTexture(ID3D11Device* device, ID3D11Texture2D* texture, ID3D11ShaderResourceView** srv) {
    if (!device || !texture || !srv) return E_INVALIDARG;
    D3D11_TEXTURE2D_DESC texDesc;
    texture->GetDesc(&texDesc);
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = texDesc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = texDesc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;
    HRESULT hr = device->CreateShaderResourceView(texture, &srvDesc, srv);
    if (FAILED(hr)) {
        std::cerr << "Failed to create SRV. Error Code: " << hr << std::endl;
    }
    return hr;
}

HRESULT CompileShader(LPCWSTR srcFile, LPCSTR entryPoint, LPCSTR profile, ID3DBlob** blob) {
    DWORD shaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
    ID3DBlob* errorBlob = nullptr;
    HRESULT hr = D3DCompileFromFile(srcFile, nullptr, nullptr, entryPoint, profile, shaderFlags, 0, blob, &errorBlob);

    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Shader Compilation Failed: " << static_cast<char*>(errorBlob->GetBufferPointer()) << " Error Code: " << hr << std::endl;
            errorBlob->Release();
        }
        return hr;
    }
    return S_OK;
}

HRESULT CreateBuffer(
    ID3D11Device* device,
    const D3D11_BUFFER_DESC& desc,
    const void* initData,
    ID3D11Buffer** buffer
) {
    D3D11_SUBRESOURCE_DATA subresourceData = {};
    subresourceData.pSysMem = initData;

    return device->CreateBuffer(&desc, initData ? &subresourceData : nullptr, buffer);
}

HRESULT CreateUAV(
    ID3D11Device* device,
    ID3D11Buffer* buffer,
    ID3D11UnorderedAccessView** uav
) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = 1;

    return device->CreateUnorderedAccessView(buffer, &uavDesc, uav);
}

HRESULT CreateReadbackBuffer(
    ID3D11Device* device,
    size_t size,
    ID3D11Buffer** buffer
) {
    D3D11_BUFFER_DESC desc = {};
    desc.Usage = D3D11_USAGE_STAGING;
    desc.ByteWidth = static_cast<uint32_t>(size);
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    return device->CreateBuffer(&desc, nullptr, buffer);
}

HRESULT SetupSamplerState(
    ID3D11Device* device,
    ID3D11SamplerState** samplerState
) {
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

    return device->CreateSamplerState(&samplerDesc, samplerState);
}

void InitializeComputeShader() {
    HRESULT hr;
    ID3DBlob* csBlob = nullptr;

    hr = CompileShader(L"ComputeShader.hlsl", "ScanTexture", "cs_5_0", &csBlob);
    if (FAILED(hr)) {
        std::cerr << "Failed to compile compute shader. Error Code: " << hr << std::endl;
        return;
    }

    hr = d3dDevice->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &ScanComputeShader);
    csBlob->Release();
    if (FAILED(hr)) {
        std::cerr << "Failed to create compute shader. Error Code: " << hr << std::endl;
        return;
    }

    hr = CompileShader(L"ComputeShader.hlsl", "CopyTexture", "cs_5_0", &csBlob);
    if (FAILED(hr)) {
        std::cerr << "Failed to compile compute shader. Error Code: " << hr << std::endl;
        return;
    }

    hr = d3dDevice->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &CopyComputeShader);
    csBlob->Release();
    if (FAILED(hr)) {
        std::cerr << "Failed to create compute shader. Error Code: " << hr << std::endl;
        return;
    }

    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    bufferDesc.ByteWidth = sizeof(uint32_t);
    bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bufferDesc.StructureByteStride = sizeof(uint32_t);
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
  
    hr = CreateBuffer(d3dDevice, bufferDesc, &xOfWindow, &xBuffer);
    hr = CreateBuffer(d3dDevice, bufferDesc, &yOfWindow, &yBuffer);
    hr = CreateUAV(d3dDevice, xBuffer, &xUAV);
    hr = CreateUAV(d3dDevice, yBuffer, &yUAV);
    hr = CreateReadbackBuffer(d3dDevice, sizeof(uint32_t), &xReadback);
    hr = CreateReadbackBuffer(d3dDevice, sizeof(uint32_t), &yReadback);

    hr = CreateBuffer(d3dDevice, bufferDesc, &imgsz, &regionXBuffer);
    hr = CreateBuffer(d3dDevice, bufferDesc, &imgsz, &regionYBuffer);
    hr = CreateUAV(d3dDevice, regionXBuffer, &regionXUAV);
    hr = CreateUAV(d3dDevice, regionYBuffer, &regionYUAV);

    hr = CreateBuffer(d3dDevice, bufferDesc, &debug, &debugBuffer);
    hr = CreateUAV(d3dDevice, debugBuffer, &debugUAV);

    hr = SetupSamplerState(d3dDevice, &samplerState);

    cudaError_t err = cudaGraphicsD3D11RegisterResource(&cudaResource, cudaTexture, cudaGraphicsRegisterFlagsNone);
    if (err != cudaSuccess) {
    }
}

void ScanTexture(ID3D11Texture2D* texture) {
    HRESULT hr;

    ID3D11ShaderResourceView* nullSRV = nullptr;
    ID3D11UnorderedAccessView* nullUAV = nullptr;
    ID3D11SamplerState* nullSampler = nullptr;
    
    uint32_t x = 0xFFFFFFFF;
    uint32_t y = 0xFFFFFFFF;
    d3dContext->UpdateSubresource(xBuffer, 0, nullptr, &x, 0, 0);
    d3dContext->UpdateSubresource(yBuffer, 0, nullptr, &y, 0, 0);

    d3dContext->UpdateSubresource(regionXBuffer, 0, nullptr, &imgsz, 0, 0);
    d3dContext->UpdateSubresource(regionYBuffer, 0, nullptr, &imgsz, 0, 0);

    d3dContext->UpdateSubresource(debugBuffer, 0, nullptr, &debug, 0, 0);

    hr = CreateSRVFromTexture(d3dDevice, desktopTexture, &desktopShaderResourceView);
    hr = CreateUAVFromTexture(d3dDevice, cudaTexture, &cudaTextureUAV);
    hr = CreateUAVFromTexture(d3dDevice, outputTexture, &outputTextureUAV);
    
    d3dContext->CSSetShader(ScanComputeShader, nullptr, 0);

    d3dContext->CSSetShaderResources(0, 1, &desktopShaderResourceView);
    d3dContext->CSSetUnorderedAccessViews(0, 1, &xUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(1, 1, &yUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(2, 1, &regionXUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(3, 1, &regionYUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(4, 1, &cudaTextureUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(5, 1, &outputTextureUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(6, 1, &debugUAV, nullptr);

    d3dContext->CSSetSamplers(0, 1, &samplerState);

    unsigned int dispatchX = (monitor_width + 31) / 32;
    unsigned int dispatchY = (monitor_height + 31) / 32;
    d3dContext->Dispatch(dispatchX, dispatchY, 1);

    d3dContext->CopyResource(xReadback, xBuffer);
    d3dContext->CopyResource(yReadback, yBuffer);

    d3dContext->CSSetShaderResources(0, 1, &nullSRV);
    d3dContext->CSSetUnorderedAccessViews(0, 1, &nullUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(1, 1, &nullUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(2, 1, &nullUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(3, 1, &nullUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(4, 1, &nullUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(5, 1, &nullUAV, nullptr);
    d3dContext->CSSetUnorderedAccessViews(6, 1, &nullUAV, nullptr);

    D3D11_MAPPED_SUBRESOURCE mappedResourceX;
    D3D11_MAPPED_SUBRESOURCE mappedResourceY;
    hr = d3dContext->Map(xReadback, 0, D3D11_MAP_READ, 0, &mappedResourceX);
    hr = d3dContext->Map(yReadback, 0, D3D11_MAP_READ, 0, &mappedResourceY);
    if (FAILED(hr)) {
        std::cerr << "Failed to map readback buffer. Error Code: " << hr << std::endl;
    }

    xOfWindow = *reinterpret_cast<uint32_t*>(mappedResourceX.pData);
    yOfWindow = *reinterpret_cast<uint32_t*>(mappedResourceY.pData);
    
    d3dContext->Unmap(xReadback, 0);
    d3dContext->Unmap(yReadback, 0);

    if (xOfWindow == 0xFFFFFFFF || yOfWindow == 0xFFFFFFFF) {
        xOfWindow = -1;
        yOfWindow = -1;
    }
    else {
        d3dContext->CSSetShader(CopyComputeShader, nullptr, 0);

        d3dContext->CSSetShaderResources(0, 1, &desktopShaderResourceView);
        d3dContext->CSSetUnorderedAccessViews(0, 1, &xUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(1, 1, &yUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(2, 1, &regionXUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(3, 1, &regionYUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(4, 1, &cudaTextureUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(5, 1, &outputTextureUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(6, 1, &debugUAV, nullptr);

        dispatchX = (monitor_width + 31) / 32;
        dispatchY = (monitor_height + 31) / 32;
        d3dContext->Dispatch(dispatchX, dispatchY, 1);

        d3dContext->CSSetShaderResources(0, 1, &nullSRV);
        d3dContext->CSSetUnorderedAccessViews(0, 1, &nullUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(1, 1, &nullUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(2, 1, &nullUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(3, 1, &nullUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(4, 1, &nullUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(5, 1, &nullUAV, nullptr);
        d3dContext->CSSetUnorderedAccessViews(6, 1, &nullUAV, nullptr);
    }
    desktopShaderResourceView->Release();
    cudaTextureUAV->Release();
    outputTextureUAV->Release();
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