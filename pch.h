#pragma once
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <shobjidl.h>

// Windows Headers
#include <windows.h>
#include <Unknwn.h>
#include <wincodec.h>
#include <dwmapi.h>


// Direct3D Headers
#include <d3dcompiler.h>
#include <dcomp.h>
//#include <winrt/windows.graphics.directx.direct3d11.h>
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
#include <d2d1_1.h>
#include <d2d1_2.h>
#include <d2d1_3.h>

// WRL Headers
#include <wrl.h>
#include <winrt/base.h>

// WinRT Headers
//#include <winrt/Windows.Graphics.Capture.h>
//#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
//#include <winrt/Windows.Graphics.DirectX.h>
//#include <windows.graphics.directx.direct3d11.interop.h>
//#include <windows.graphics.capture.h>
//#include <Windows.Graphics.Capture.Interop.h>
//#include <winrt/Windows.Foundation.Metadata.h>
//#include <winrt/Windows.Foundation.h>
//#include <winrt/Windows.Foundation.Collections.h>
