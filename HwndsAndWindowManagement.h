#pragma once
#include "globals.h"

extern inline void GetClientAreaInBox() {
    RECT clientRect;

    if (GetClientRect(lookingGlassHwnd, &clientRect)) {
        POINT topLeft = { clientRect.left, clientRect.top };
        POINT bottomRight = { clientRect.right, clientRect.bottom };
        ClientToScreen(lookingGlassHwnd, &topLeft);
        ClientToScreen(lookingGlassHwnd, &bottomRight);

        srcBox.left = topLeft.x;
        srcBox.top = topLeft.y;
        srcBox.right = bottomRight.x;
        srcBox.bottom = bottomRight.y;
        srcBox.front = 0;
        srcBox.back = 1;
    }
    else {
        srcBox.left = srcBox.top = srcBox.right = srcBox.bottom = 0;
        srcBox.front = 0;
        srcBox.back = 1;
    }
}

void GetScreenSizeFromHMonitor(HMONITOR hMonitor, int& width, int& height) {
    MONITORINFO monitorInfo = { sizeof(MONITORINFO) };
    if (GetMonitorInfo(hMonitor, &monitorInfo)) {
        width = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
        height = monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top;
    }
    else {
        std::cerr << "Failed to get monitor info." << std::endl;
        width = 0;
        height = 0;
    }
}

HICON CreateCustomIcon() {
    const int width = 16;
    const int height = 16;

    HDC hdc = GetDC(nullptr);
    HDC hdcMem = CreateCompatibleDC(hdc);
    HBITMAP hBitmap = CreateCompatibleBitmap(hdc, width, height);
    HBITMAP hOldBitmap = (HBITMAP)SelectObject(hdcMem, hBitmap);

    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    void* pvBits = nullptr;
    HBITMAP hDIB = CreateDIBSection(hdcMem, &bmi, DIB_RGB_COLORS, &pvBits, nullptr, 0);
    SelectObject(hdcMem, hDIB);
    ZeroMemory(pvBits, width * height * 4);

    for (int y = 0; y < height; ++y) {
        COLORREF color;
        if (y % 3 == 0) {
            color = RGB(255, 0, 0);
        }
        else if (y % 3 == 1) {
            color = RGB(0, 255, 0);
        }
        else {
            color = RGB(0, 0, 255);
        }
        SetPixel(hdcMem, 0, y, color);
    }

    ICONINFO iconInfo = {};
    iconInfo.fIcon = TRUE;
    iconInfo.xHotspot = 0;
    iconInfo.yHotspot = 0;
    iconInfo.hbmMask = hDIB;
    iconInfo.hbmColor = hDIB;
    HICON hIcon = CreateIconIndirect(&iconInfo);

    SelectObject(hdcMem, hOldBitmap);
    DeleteObject(hDIB);
    DeleteDC(hdcMem);
    ReleaseDC(nullptr, hdc);

    return hIcon;
}

LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message)
    {
    case WM_WINDOWPOSCHANGING:
    {
        LPWINDOWPOS lpwp = reinterpret_cast<LPWINDOWPOS>(lParam);
        if ((lpwp->flags & SWP_NOMOVE) != 0) 
        {
            break;
        }

        RECT workArea = {0};
        SystemParametersInfo(SPI_GETWORKAREA, 0, &workArea, 0);

        int width = lpwp->cx;
        int height = lpwp->cy;

        if (lpwp->x < workArea.left)
        {
            lpwp->x = workArea.left;
        }
        else if (lpwp->x + width > workArea.right)
        {
            lpwp->x = workArea.right - width;
        }

        if (lpwp->y < workArea.top)
        {
            lpwp->y = workArea.top;
        }
        else if (lpwp->y + height > workArea.bottom)
        {
            lpwp->y = workArea.bottom - height;
        }
        xOfWindow = lpwp->x;
        yOfWindow = lpwp->y;
    }
    break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}


void CreateOverlayAndLookingGlass() {
    WNDCLASS wc = {};
    wc.lpfnWndProc = DefWindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "overlay";
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = NULL;
    wc.style = CS_HREDRAW | CS_VREDRAW;

    ATOM classAtom = RegisterClass(&wc);
    assert(classAtom != 0);

    DWORD dwStyle = WS_POPUP | WS_VISIBLE;
    DWORD dwExStyle = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_NOREDIRECTIONBITMAP;

    overlayHwnd = CreateWindowExW(
        dwExStyle,
        L"overlay",
        L"",
        dwStyle,
        0, 0, monitor_width, monitor_height,
        NULL,
        NULL,
        hInstance,
        NULL
    );
    assert(overlayHwnd != NULL);

    SetWindowDisplayAffinity(overlayHwnd, WDA_EXCLUDEFROMCAPTURE);
    SetLayeredWindowAttributes(overlayHwnd, 0, 255, LWA_ALPHA);

    // Register Looking Glass Window
    WNDCLASS wndClass = {};
    wndClass.lpfnWndProc = WindowProc;
    wndClass.hInstance = hInstance;
    wndClass.hbrBackground = NULL;
    wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wndClass.lpszClassName = "LookingGlass";
    wndClass.style = CS_HREDRAW | CS_VREDRAW;

    ATOM classAtoml = RegisterClass(&wndClass);
    assert(classAtoml != 0);

    dwStyle = (WS_OVERLAPPEDWINDOW & ~WS_MAXIMIZEBOX) | WS_VISIBLE;

    lookingGlassHwnd = CreateWindowEx(
        WS_EX_NOREDIRECTIONBITMAP,
        "LookingGlass",  // Matching the class name with registration
        "Looking Glass",
        dwStyle,
        0, 0,
        imgsz, imgsz,
        NULL,
        NULL,
        hInstance,
        NULL
    );
    assert(lookingGlassHwnd != NULL);

    HICON hIcon = CreateCustomIcon();
    SendMessage(lookingGlassHwnd, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);

    ShowWindow(lookingGlassHwnd, SW_SHOWDEFAULT);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}





