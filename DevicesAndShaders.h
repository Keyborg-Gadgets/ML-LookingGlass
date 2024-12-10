#ifndef DIRECTRENDER_H
#define DIRECTRENDER_H
#define COBJMACROS

#include <windows.h>
#include <d3d11_1.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <dxgi1_3.h>
#include <dxgi1_4.h>
#include <dxgi1_5.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <d3d11_2.h>
#include <d3d11_3.h>
#include <d3d11_4.h>
#include <d2d1_2.h>
#include <dxgi1_2.h>
#include <d3d11.h>
#include <assert.h>
#include <string.h>

#include <d2d1_3.h>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif
    typedef struct Constants {
        int frame;
    } Constants;

    void CreateDeviceAndContext(
        IDXGIAdapter1* adapter,
        ID3D11Device1** device,
        ID3D11DeviceContext1** ctx,
        IDXGIFactory2** d3dfactory
    );

    IDXGISwapChain1* CreateSwapChain(ID3D11Device1* device, int width, int height);

    void FinalSwap(IDXGISwapChain1* swapchain);

    //void CreateD2DDevice(
    //    IDXGIDevice1* dxgiDevice,
    //    IDXGISwapChain1* swapchain,
    //    ID2D1Factory2** d2dFactory,
    //    ID2D1Device1** d2dDevice,
    //    ID2D1DeviceContext** d2dContext,
    //    IDXGISurface2** dxgiBackBuffer,
    //    ID2D1Bitmap1** d2dBitmapBackBuffer
    //);

    void Cleanup(
        ID3D11DeviceContext1** ctx,
        ID3D11Device1** device,
        IDXGISwapChain1** swapchain,
        ID3D11RenderTargetView** rtv,
        ID3D11Texture2D** texture,
        ID3D11Texture2D** backBuffer
    );

    IDXGIAdapter1* GetDefaultAdapter();
    IDXGIDevice1* GetDxgiDevice(ID3D11Device1* device);

    void CreateRenderTargetView(ID3D11Device1* device, IDXGISwapChain1* swapchain, ID3D11RenderTargetView** rtv, ID3D11Texture2D** backBuffer);
    void RenderTextureToBackBuffer(ID3D11DeviceContext1* ctx, ID3D11Texture2D* texture, ID3D11Texture2D* backBuffer);

#ifdef __cplusplus
}
#endif

#endif

