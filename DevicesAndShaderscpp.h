#pragma once
#include "globals.h"
#include "DevicesAndShaders.h"

inline void CreateD2DDevice(
    IDXGIDevice1* dxgiDevice,
    IDXGISwapChain1* swapchain,
    ID2D1Factory2** d2dFactory,
    ID2D1Device1** d2dDevice,
    ID2D1DeviceContext** d2dContext,
    IDXGISurface2** dxgiBackBuffer,
    ID2D1Bitmap1** d2dBitmapBackBuffer
) {
    HRESULT hr;
    D2D1_FACTORY_OPTIONS options;
    memset(&options, 0, sizeof(options));

#if defined(_DEBUG)
    options.debugLevel = D2D1_DEBUG_LEVEL_INFORMATION;
#endif

    // Create D2D1 Factory
    hr = D2D1CreateFactory(
        D2D1_FACTORY_TYPE_SINGLE_THREADED,
        __uuidof(ID2D1Factory2),
        &options,
        reinterpret_cast<void**>(d2dFactory)
    );
    assert(SUCCEEDED(hr) && "Failed to create D2D1 factory");

    // Create D2D Device from the DXGIDevice
    hr = (*d2dFactory)->CreateDevice(reinterpret_cast<IDXGIDevice*>(dxgiDevice), d2dDevice);
    assert(SUCCEEDED(hr) && "Failed to create D2D1 device");

    // Create D2D Device Context
    hr = (*d2dDevice)->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, d2dContext);
    assert(SUCCEEDED(hr) && "Failed to create D2D1 device context");

    // Get the back buffer as IDXGISurface from the swap chain
    hr = swapchain->GetBuffer(0, __uuidof(IDXGISurface2), reinterpret_cast<void**>(dxgiBackBuffer));
    assert(SUCCEEDED(hr) && "Failed to get back buffer from swap chain");

    // Set bitmap properties
    D2D1_BITMAP_PROPERTIES1 bitmapProperties;
    memset(&bitmapProperties, 0, sizeof(bitmapProperties));
    bitmapProperties.pixelFormat.format = DXGI_FORMAT_B8G8R8A8_UNORM;
    bitmapProperties.pixelFormat.alphaMode = D2D1_ALPHA_MODE_PREMULTIPLIED;
    bitmapProperties.bitmapOptions = D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW;

    // Create a D2D1 bitmap from the DXGI surface
    hr = (*d2dContext)->CreateBitmapFromDxgiSurface(*dxgiBackBuffer, &bitmapProperties, d2dBitmapBackBuffer);
    assert(SUCCEEDED(hr) && "Failed to create D2D1 bitmap from DXGI surface");

    // Set the render target of the D2D device context
    (*d2dContext)->SetTarget(*d2dBitmapBackBuffer);
}

struct uint2 {
    unsigned int x;
    unsigned int y;
};

struct __declspec(uuid("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1"))
    IDirect3DDxgiInterfaceAccess : ::IUnknown
{
    virtual HRESULT __stdcall GetInterface(GUID const& id, void** object) = 0;
};

template <typename T>
auto GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const& object)
{
    auto access = object.as<IDirect3DDxgiInterfaceAccess>();
    winrt::com_ptr<T> result;
    winrt::check_hresult(access->GetInterface(winrt::guid_of<T>(), result.put_void()));
    return result;
}

inline HRESULT CompileShader(LPCWSTR srcFile, LPCSTR entryPoint, LPCSTR profile, ID3DBlob** blob) {
    return D3DCompileFromFile(srcFile, nullptr, nullptr, entryPoint, profile, D3DCOMPILE_ENABLE_STRICTNESS, 0, blob, nullptr);
}

inline void CreateDCompDevice() {
    IDXGIDevice* dxgiDevice = nullptr;
    HRESULT hr = d3dDevice->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice));
    assert(SUCCEEDED(hr) && "Failed to get IDXGIDevice from ID3D11Device1");

    hr = DCompositionCreateDevice(dxgiDevice, __uuidof(IDCompositionDevice), reinterpret_cast<void**>(&dcompDevice));
    dxgiDevice->Release();
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

// Function to render using Direct2D and DirectComposition
inline void Render() {
    d2dContext->BeginDraw();

    d2dContext->Clear(D2D1::ColorF(D2D1::ColorF::Black, 0.0f));

    // Create a solid color brush
    ID2D1SolidColorBrush* brush = nullptr;
    D2D1_COLOR_F brushColor = D2D1::ColorF(0.18f, 0.55f, 0.34f, 0.75f);
    HRESULT hr = d2dContext->CreateSolidColorBrush(brushColor, &brush);
    assert(SUCCEEDED(hr) && "Failed to create solid color brush");

    // Draw an ellipse
    D2D1_POINT_2F ellipseCenter = D2D1::Point2F(150.0f, 150.0f);
    D2D1_ELLIPSE ellipse = D2D1::Ellipse(ellipseCenter, 100.0f, 100.0f);
    d2dContext->FillEllipse(ellipse, brush);
    brush->Release(); // Release the brush after use

    hr = d2dContext->EndDraw();
    assert(SUCCEEDED(hr) && "Failed to end D2D drawing");

    // Present the swap chain
    hr = swapchain->Present(1, 0);
    assert(SUCCEEDED(hr) && "Failed to present swap chain");

    // Commit the DirectComposition device
    hr = dcompDevice->Commit();
    assert(SUCCEEDED(hr) && "Failed to commit DComposition device");
}

// Function to scan the texture using a compute shader
inline std::vector<unsigned int> ScanTexture(ID3D11Texture2D* inputTexture) {
    // Initialize output buffer with default values
    uint2 initialValue = { 0xFFFFFFFF, 0xFFFFFFFF };
    d3dContext->UpdateSubresource(outputBuffer, 0, nullptr, &initialValue, 0, 0);

    // Describe the input texture
    D3D11_TEXTURE2D_DESC desc;
    inputTexture->GetDesc(&desc);
    if (desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM) return {};

    // Create shader resource view for the input texture
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = desc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;

    ID3D11ShaderResourceView* srv = nullptr;
    HRESULT hr = d3dDevice->CreateShaderResourceView(inputTexture, &srvDesc, &srv);
    if (FAILED(hr)) return {};

    // Set compute shader and resources
    d3dContext->CSSetShader(computeShader, nullptr, 0);
    d3dContext->CSSetShaderResources(0, 1, &srv);
    d3dContext->CSSetUnorderedAccessViews(0, 1, &outputBufferUAV, nullptr);
    d3dContext->CSSetSamplers(0, 1, &samplerState);

    // Dispatch compute shader
    d3dContext->Dispatch((desc.Width + 15) / 16, (desc.Height + 15) / 16, 1);

    // Unbind resources
    ID3D11ShaderResourceView* nullSRV = nullptr;
    ID3D11UnorderedAccessView* nullUAV = nullptr;
    d3dContext->CSSetShaderResources(0, 1, &nullSRV);
    d3dContext->CSSetUnorderedAccessViews(0, 1, &nullUAV, nullptr);
    ID3D11SamplerState* nullSampler = nullptr;
    d3dContext->CSSetSamplers(0, 1, &nullSampler);

    // Copy output buffer to readback buffer
    d3dContext->CopyResource(outputBufferReadback, outputBuffer);

    // Map the readback buffer to access the results
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    hr = d3dContext->Map(outputBufferReadback, 0, D3D11_MAP_READ, 0, &mappedResource);
    if (FAILED(hr)) {
        srv->Release();
        return {};
    }

    // Retrieve the position data
    uint2* outputData = reinterpret_cast<uint2*>(mappedResource.pData);
    uint2 position = outputData[0];
    d3dContext->Unmap(outputBufferReadback, 0);

    srv->Release();

    // Check if a valid position was found
    if (position.x != 0xFFFFFFFF && position.y != 0xFFFFFFFF)
        return { position.x, position.y };
    return {};
}

// Function to clean up compute shader resources
inline void CleanupComputeShader() {
    if (computeShader) {
        computeShader->Release();
        computeShader = nullptr;
    }
    if (outputBuffer) {
        outputBuffer->Release();
        outputBuffer = nullptr;
    }
    if (outputBufferUAV) {
        outputBufferUAV->Release();
        outputBufferUAV = nullptr;
    }
    if (outputBufferReadback) {
        outputBufferReadback->Release();
        outputBufferReadback = nullptr;
    }
    if (samplerState) {
        samplerState->Release();
        samplerState = nullptr;
    }
}

//inline auto CreateDirect3DDevice(ID3D11Device1* d3dDevice) {
//    if (d3dDevice == nullptr) {
//        throw winrt::hresult_error(E_INVALIDARG, L"d3dDevice is null.");
//    }
//
//    HRESULT hr =
//        d3dDevice->QueryInterface(__uuidof(IDXGIDevice), dxgiDevice.put_void());
//    if (FAILED(hr)) {
//        throw winrt::hresult_error(hr,
//            L"Failed to get IDXGIDevice from ID3D11Device.");
//    }
//
//    winrt::com_ptr<IInspectable> inspectableDevice;
//    hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.get(),
//        inspectableDevice.put());
//    if (FAILED(hr)) {
//        throw winrt::hresult_error(
//            hr, L"Failed to create IDirect3DDevice from IDXGIDevice.");
//    }
//
//    direct3DDevice = inspectableDevice.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
//}

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
    if (FAILED(hr))
        throw winrt::hresult_error(hr, L"Failed to create 2D texture.");
}

// Function to initialize the compute shader and related resources
inline void InitializeComputeShader() {
    ID3DBlob* csBlob = nullptr;
    if (FAILED(CompileShader(L"ComputeShader.hlsl", "CSMain", "cs_5_0", &csBlob))) {
        // Handle shader compilation failure
        return;
    }

    HRESULT hr = d3dDevice->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &computeShader);
    csBlob->Release();
    if (FAILED(hr)) {
        // Handle compute shader creation failure
        return;
    }

    // Describe the output buffer
    D3D11_BUFFER_DESC outputBufferDesc = {};
    outputBufferDesc.ByteWidth = sizeof(uint2);
    outputBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    outputBufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    outputBufferDesc.CPUAccessFlags = 0;
    outputBufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    outputBufferDesc.StructureByteStride = sizeof(uint2);

    // Initialize the output buffer with default values
    uint2 initialValue = { 0xFFFFFFFF, 0xFFFFFFFF };
    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = &initialValue;

    hr = d3dDevice->CreateBuffer(&outputBufferDesc, &initData, &outputBuffer);
    if (FAILED(hr)) {
        CleanupComputeShader();
        return;
    }

    // Describe the unordered access view (UAV) for the output buffer
    D3D11_UNORDERED_ACCESS_VIEW_DESC outputUAVDesc = {};
    outputUAVDesc.Format = DXGI_FORMAT_UNKNOWN;
    outputUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    outputUAVDesc.Buffer.FirstElement = 0;
    outputUAVDesc.Buffer.NumElements = 1;

    hr = d3dDevice->CreateUnorderedAccessView(outputBuffer, &outputUAVDesc, &outputBufferUAV);
    if (FAILED(hr)) {
        CleanupComputeShader();
        return;
    }

    // Describe the readback buffer for retrieving results
    D3D11_BUFFER_DESC readbackBufferDesc = {};
    readbackBufferDesc.ByteWidth = sizeof(uint2);
    readbackBufferDesc.Usage = D3D11_USAGE_STAGING;
    readbackBufferDesc.BindFlags = 0;
    readbackBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    readbackBufferDesc.MiscFlags = 0;

    hr = d3dDevice->CreateBuffer(&readbackBufferDesc, nullptr, &outputBufferReadback);
    if (FAILED(hr)) {
        CleanupComputeShader();
        return;
    }

    // Describe the sampler state
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.MipLODBias = 0.0f;
    samplerDesc.MaxAnisotropy = 0;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
    samplerDesc.BorderColor[0] = 0.0f;
    samplerDesc.BorderColor[1] = 0.0f;
    samplerDesc.BorderColor[2] = 0.0f;
    samplerDesc.BorderColor[3] = 0.0f;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

    hr = d3dDevice->CreateSamplerState(&samplerDesc, &samplerState);
    if (FAILED(hr)) {
        CleanupComputeShader();
    }
}