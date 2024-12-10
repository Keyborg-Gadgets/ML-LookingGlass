#pragma once
#include "globals.h"

// Function to get the size of the screen from an HMONITOR
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

void CreateOverlay() {
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
    DWORD dwExStyle = WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_NOREDIRECTIONBITMAP;

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

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

