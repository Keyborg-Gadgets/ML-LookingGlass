#pragma once
#include "globals.h"
#include <windows.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")

#include <windows.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")

// Function to darken the color
COLORREF DarkenColor(COLORREF color, double percentage) {
    BYTE r = GetRValue(color);
    BYTE g = GetGValue(color);
    BYTE b = GetBValue(color);

    r = static_cast<BYTE>(r * (1 - percentage));
    g = static_cast<BYTE>(g * (1 - percentage));
    b = static_cast<BYTE>(b * (1 - percentage));

    return RGB(r, g, b);
}



// Function to set the title bar color
void SetTitleBarColor(HWND hwnd) {
    COLORREF currentColor = GetSysColor(COLOR_ACTIVECAPTION);
    COLORREF darkerColor = DarkenColor(currentColor, 0.17);
    COLORREF DARK_COLOR = 0x00505050;
    DwmSetWindowAttribute(
        hwnd, DWMWINDOWATTRIBUTE::DWMWA_CAPTION_COLOR,
        &DARK_COLOR, sizeof(DARK_COLOR));
}



void SetStyles(HWND hwnd)
{
    if (!IsWindow(hwnd)) {
        return; 
    }

    LONG_PTR exStyle = GetWindowLongPtr(hwnd, GWL_EXSTYLE);
    if (exStyle == 0) {
        return; 
    }
    exStyle &= ~WS_MAXIMIZEBOX | ~WS_THICKFRAME | WS_CAPTION;
    SetWindowLongPtr(hwnd, GWL_EXSTYLE, exStyle);

    LONG style = GetWindowLong(hwnd, GWL_STYLE);
    style &= ~WS_MINIMIZEBOX;
    SetWindowLong(hwnd, GWL_STYLE, style);

    // despite my best efforts the client region is off by 1 for each border.
    RECT rect = { 0, 0, imgsz + 2, imgsz + 1};
    AdjustWindowRect(&rect, style, FALSE);

    HMENU hMenu = GetSystemMenu(hwnd, FALSE);
    EnableMenuItem(hMenu, SC_MINIMIZE, MF_BYCOMMAND | MF_GRAYED);
    SetWindowPos(hwnd, nullptr, 0, 0, rect.right - rect.left, rect.bottom - rect.top,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
    SetTitleBarColor(hwnd);
}


void RemoveFromTaskbar(HWND hwnd)
{
    CoInitialize(NULL);

    ITaskbarList* pTaskbarList = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_TaskbarList, nullptr, CLSCTX_INPROC_SERVER,
        IID_ITaskbarList, reinterpret_cast<void**>(&pTaskbarList));
    if (SUCCEEDED(hr) && pTaskbarList)
    {
        hr = pTaskbarList->HrInit();
        if (SUCCEEDED(hr))
        {
            pTaskbarList->DeleteTab(hwnd);
        }
        pTaskbarList->Release();
    }

    CoUninitialize();
}

inline void GetTitleBarHeight(HWND hwnd) {
    TITLEBARINFOEX* ptinfo = (TITLEBARINFOEX*)malloc(sizeof(TITLEBARINFOEX));
    ptinfo->cbSize = sizeof(TITLEBARINFOEX);
    SendMessage(hwnd, WM_GETTITLEBARINFOEX, 0, (LPARAM)ptinfo);
    titleBarHeight = ptinfo->rcTitleBar.bottom - ptinfo->rcTitleBar.top;
    free(ptinfo);
}

void GetScreenSizeFromHMonitor(HMONITOR hMonitor, unsigned int& width, unsigned int& height) {
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
        for (int x = 0; x < width; ++x) {
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
            SetPixel(hdcMem, x, y, color);
        }
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
    }
    break;

    case WM_DESTROY:
        PostQuitMessage(0);
        exit(0);
        break;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

void MakeInteractive(HWND hwnd, bool makeInteractive) {
    LONG_PTR exStyle = GetWindowLongPtr(hwnd, GWL_EXSTYLE);

    if (makeInteractive) {
        // Remove the transparent and no-activate flags
        exStyle &= ~(WS_EX_TRANSPARENT | WS_EX_NOACTIVATE);
    }
    else {
        // Add the transparent and no-activate flags
        exStyle |= (WS_EX_TRANSPARENT | WS_EX_NOACTIVATE);
    }

    SetWindowLongPtr(hwnd, GWL_EXSTYLE, exStyle);

    // To ensure the changes take effect immediately:
    SetWindowPos(hwnd, NULL, 0, 0, 0, 0,
        SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED);
}

#include <iostream>

// Clamp function
unsigned int clamp(unsigned int value, unsigned int min, unsigned int max) {
    if (value < min) {
        return min;
    }
    else if (value > max) {
        return max;
    }
    else {
        return value;
    }
}


#undef max
unsigned int addWithClamp(unsigned int value, int numberToAdd) {
    // If numberToAdd is negative and its absolute value is greater than value, return 0
    if (numberToAdd < 0 && static_cast<unsigned int>(-numberToAdd) > value) {
        return 0;
    }

    unsigned int result = value + numberToAdd;

    if (numberToAdd > 0 && result < value) {
        // Overflow occurred
        return std::numeric_limits<unsigned int>::max();
    }

    return result;
}
int statechecked = 0;
LRESULT CALLBACK LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    static bool previouslyInteractive = false;
    bool windowAvailable = (xOfWindow != -1 && yOfWindow != -1);
    bool inBounds = false;
    bool interactive = true;

    if (nCode == HC_ACTION)
    {
        PMSLLHOOKSTRUCT pMouseStruct = (PMSLLHOOKSTRUCT)lParam;
        if (pMouseStruct != nullptr)
        {
            xOfMouse = pMouseStruct->pt.x;
            yOfMouse = pMouseStruct->pt.y;
            windowAvailable = (xOfWindow != -1 && yOfWindow != -1);
            if (windowAvailable) {
                inBounds = (xOfMouse > xOfWindow && xOfMouse < xOfWindow + imgsz &&
                    yOfMouse < yOfWindow + 4 && yOfMouse > addWithClamp(yOfWindow, -40));
            }

            if (inBounds) {
                statechecked = statechecked + 1;
            }
            else {
                statechecked = statechecked - 1;
            }

            interactive = windowAvailable && inBounds;
            if (interactive != previouslyInteractive && (statechecked >= 1 || statechecked <= -5)) {
                statechecked = 0;
                MakeInteractive(lookingGlassHwnd, interactive);
                previouslyInteractive = interactive;
            }
        }
    }
    return CallNextHookEx(hMouseHook, nCode, wParam, lParam);
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

    
    WNDCLASS wndClass = {};
    wndClass.lpfnWndProc = WindowProc;
    wndClass.hInstance = hInstance;
    wndClass.hbrBackground = NULL;
    wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wndClass.lpszClassName = "LookingGlass";
    wndClass.style = CS_HREDRAW | CS_VREDRAW;

    ATOM classAtoml = RegisterClass(&wndClass);
    assert(classAtoml != 0);

    dwStyle = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    dwExStyle = WS_EX_OVERLAPPEDWINDOW | WS_EX_NOREDIRECTIONBITMAP | WS_EX_TOPMOST | WS_EX_LAYERED;

    RECT rect = { 0, 0, static_cast<LONG>(imgsz), static_cast<LONG>(imgsz) };
    AdjustWindowRectEx(&rect, dwStyle, FALSE, dwExStyle);

    int x = (monitor_width - imgsz) / 2;
    int y = (monitor_height - imgsz) / 2;

    lookingGlassHwnd = CreateWindowEx(
        dwExStyle,
        "LookingGlass",  
        "Looking Glass",
        dwStyle,
        x, y,
        rect.right - rect.left,
        rect.bottom - rect.top,
        NULL,
        NULL,
        hInstance,
        NULL
    );
    assert(lookingGlassHwnd != NULL);
    GetTitleBarHeight(lookingGlassHwnd);

    HICON hIcon = CreateCustomIcon();
    SendMessage(lookingGlassHwnd, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);



    SetStyles(lookingGlassHwnd);

    RemoveFromTaskbar(lookingGlassHwnd);
    SetWindowsHookEx(WH_MOUSE_LL, LowLevelMouseProc, GetModuleHandle(NULL), 0);
    SetLayeredWindowAttributes(lookingGlassHwnd, 0, 255, LWA_ALPHA);
    ShowWindow(lookingGlassHwnd, SW_SHOWDEFAULT);
    winDone = true;


 
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}