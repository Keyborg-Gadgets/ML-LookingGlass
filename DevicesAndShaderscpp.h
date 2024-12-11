#pragma once
#include "globals.h"
#include "DevicesAndShaders.h"

struct uint2 {
    unsigned int x;
    unsigned int y;
};

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

HRESULT CopyDirtyRects() {
    UINT requiredSize = 0;
    hr = outputDuplication->GetFrameDirtyRects(0, nullptr, &requiredSize);
    if (hr == DXGI_ERROR_MORE_DATA){}
    else if (FAILED(hr))
    {
        return hr;
    }
    if (requiredSize == 0)
    {
        return S_OK;
    }

    UINT numRects = requiredSize / sizeof(RECT);

    std::vector<RECT> dirtyRects(numRects);

    hr = outputDuplication->GetFrameDirtyRects(requiredSize, dirtyRects.data(), &requiredSize);
    if (FAILED(hr))
    {
        return hr;
    }

    D3D11_TEXTURE2D_DESC desc;
    desktopTexture->GetDesc(&desc);

    for (const auto& rect : dirtyRects)
    {
        LONG left = std::max<LONG>(0, rect.left);
        LONG top = std::max<LONG>(0, rect.top);
        LONG right = std::min<LONG>(static_cast<LONG>(desc.Width), rect.right);
        LONG bottom = std::min<LONG>(static_cast<LONG>(desc.Height), rect.bottom);

        if (right <= left || bottom <= top)
        {
            continue;
        }

        D3D11_BOX srcBox;
        srcBox.left = left;
        srcBox.top = top;
        srcBox.front = 0;
        srcBox.right = right;
        srcBox.bottom = bottom;
        srcBox.back = 1;

        d3dContext->CopySubresourceRegion(
            d2dTextureBackBuffer,
            0,              
            left,           
            top,            
            0,              
            desktopTexture,
            0,              
            &srcBox
        );
    }

    return S_OK;
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

inline void Create2DTexture(ID3D11Texture2D** texture, int width, int height) {
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

    HRESULT hr = d3dDevice->CreateTexture2D(&desc, nullptr, texture);
}

inline void CreateDesktopSRV() {
    D3D11_TEXTURE2D_DESC desc;
    desktopTexture->GetDesc(&desc);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = desc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;


    hr = d3dDevice->CreateShaderResourceView(desktopTexture, &srvDesc, &desktopShaderResourceView);
    if (FAILED(hr)) {
        std::cerr << "Failed to create shader resource view. Error Code: " << hr << std::endl;
        exit(1);
    }
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

void InitializeComputeShader() {
    HRESULT hr;
    ID3DBlob* csBlob = nullptr;

    hr = CompileShader(L"ComputeShader.hlsl", "CSMain", "cs_5_0", &csBlob);
    if (FAILED(hr)) {
        std::cerr << "Failed to compile compute shader. Error Code: " << hr << std::endl;
        return;
    }

    hr = d3dDevice->CreateComputeShader(
        csBlob->GetBufferPointer(),
        csBlob->GetBufferSize(),
        nullptr,
        &computeShader
    );
    csBlob->Release();
    if (FAILED(hr)) {
        std::cerr << "Failed to create compute shader. Error Code: " << hr << std::endl;
        return;
    }

    D3D11_BUFFER_DESC outputBufferDesc = {};
    outputBufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    outputBufferDesc.ByteWidth = sizeof(uint2);
    outputBufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    outputBufferDesc.StructureByteStride = sizeof(uint2);
    outputBufferDesc.Usage = D3D11_USAGE_DEFAULT;

    uint2 initialValue = { 0xFFFFFFFF, 0xFFFFFFFF };
    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = &initialValue;

    hr = d3dDevice->CreateBuffer(&outputBufferDesc, &initData, &outputBuffer);
    if (FAILED(hr)) {
        std::cerr << "Failed to create output buffer. Error Code: " << hr << std::endl;
        computeShader->Release();
        computeShader = nullptr;
        return;
    }

    D3D11_UNORDERED_ACCESS_VIEW_DESC outputUAVDesc = {};
    outputUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    outputUAVDesc.Format = DXGI_FORMAT_UNKNOWN;
    outputUAVDesc.Buffer.FirstElement = 0;
    outputUAVDesc.Buffer.NumElements = 1;

    hr = d3dDevice->CreateUnorderedAccessView(outputBuffer, &outputUAVDesc, &outputBufferUAV);
    if (FAILED(hr)) {
        std::cerr << "Failed to create UAV for output buffer. Error Code: " << hr << std::endl;
        outputBuffer->Release();
        outputBuffer = nullptr;
        computeShader->Release();
        computeShader = nullptr;
        return;
    }

    D3D11_BUFFER_DESC readbackBufferDesc = {};
    readbackBufferDesc.Usage = D3D11_USAGE_STAGING;
    readbackBufferDesc.ByteWidth = sizeof(uint2);
    readbackBufferDesc.BindFlags = 0;
    readbackBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    readbackBufferDesc.MiscFlags = 0;

    hr = d3dDevice->CreateBuffer(&readbackBufferDesc, nullptr, &outputBufferReadback);
    if (FAILED(hr)) {
        std::cerr << "Failed to create readback buffer. Error Code: " << hr << std::endl;
        outputBufferUAV->Release();
        outputBufferUAV = nullptr;
        outputBuffer->Release();
        outputBuffer = nullptr;
        computeShader->Release();
        computeShader = nullptr;
        return;
    }

    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

    hr = d3dDevice->CreateSamplerState(&samplerDesc, &samplerState);
    if (FAILED(hr)) {
        std::cerr << "Failed to create sampler state. Error Code: " << hr << std::endl;
        
        outputBufferReadback->Release();
        outputBufferReadback = nullptr;
        outputBufferUAV->Release();
        outputBufferUAV = nullptr;
        outputBuffer->Release();
        outputBuffer = nullptr;
        computeShader->Release();
        computeShader = nullptr;
        return;
    }
}

void ScanTexture(ID3D11Texture2D* texture) {
    HRESULT hr;

    
    uint2 initialValue = { 0xFFFFFFFF, 0xFFFFFFFF };
    d3dContext->UpdateSubresource(outputBuffer, 0, nullptr, &initialValue, 0, 0);

    
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);

    
    if (desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM) {
        std::cerr << "Texture format is not DXGI_FORMAT_B8G8R8A8_UNORM." << std::endl;
    }

    
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = desc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;

    ID3D11ShaderResourceView* srv = nullptr;
    hr = d3dDevice->CreateShaderResourceView(texture, &srvDesc, &srv);
    if (FAILED(hr)) {
        std::cerr << "Failed to create shader resource view. Error Code: " << hr << std::endl;
    }

    
    d3dContext->CSSetShader(computeShader, nullptr, 0);

    
    d3dContext->CSSetShaderResources(0, 1, &srv);

    
    d3dContext->CSSetUnorderedAccessViews(0, 1, &outputBufferUAV, nullptr);

    
    d3dContext->CSSetSamplers(0, 1, &samplerState);

    
    unsigned int dispatchX = (desc.Width + 15) / 16;
    unsigned int dispatchY = (desc.Height + 15) / 16;

    
    d3dContext->Dispatch(dispatchX, dispatchY, 1);

    
    ID3D11ShaderResourceView* nullSRV = nullptr;
    ID3D11UnorderedAccessView* nullUAV = nullptr;
    ID3D11SamplerState* nullSampler = nullptr;
    d3dContext->CSSetShaderResources(0, 1, &nullSRV);
    d3dContext->CSSetUnorderedAccessViews(0, 1, &nullUAV, nullptr);
    d3dContext->CSSetSamplers(0, 1, &nullSampler);

    
    srv->Release();

    
    d3dContext->CopyResource(outputBufferReadback, outputBuffer);

    
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    hr = d3dContext->Map(outputBufferReadback, 0, D3D11_MAP_READ, 0, &mappedResource);
    if (FAILED(hr)) {
        std::cerr << "Failed to map readback buffer. Error Code: " << hr << std::endl;
    }

    
    uint2* outputData = reinterpret_cast<uint2*>(mappedResource.pData);
    uint2 position = outputData[0];

    
    d3dContext->Unmap(outputBufferReadback, 0);

    
    if (position.x != 0xFFFFFFFF && position.y != 0xFFFFFFFF) {
        
        
        xOfWindow = position.x - 10;
        yOfWindow = position.y + 10;
    }
    else {
        
        xOfWindow = -1;
        yOfWindow = -1;
    }
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