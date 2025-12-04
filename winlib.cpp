#include "winlib.h"
#include <stdlib.h>
#include <tchar.h>

// Ensure InputData is defined if not already included by winlib.h

static const char* WINLIB_CLASS_NAME = "WinLibWindowClass";

// 2. Global variable to keep track of the most up-to-date InputData.
static InputData g_inputData = {0};
// Track the last known client mouse position so we can compute deltas
static int g_lastMouseX = 0;
static int g_lastMouseY = 0;

// 3. Function to return the current input data.
InputData WinLib_GetInputs(void) {
    // 1. Pump all pending Windows messages
    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // 2. Take a snapshot of the current input state
    InputData snapshot = g_inputData;

    // 3. Reset the deltas so new events accumulate for the next call
    g_inputData.mouseX = 0;
    g_inputData.mouseY = 0;

    return snapshot;
}

/**
 * The window procedure for handling messages sent to windows created by this library.
 */
static LRESULT CALLBACK WinLib_WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        // Update mouse position on movement (raw input - relative deltas).
        case WM_INPUT: {
            UINT dwSize = 0;
            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, nullptr, &dwSize, sizeof(RAWINPUTHEADER));
            RAWINPUT* raw = (RAWINPUT*)malloc(dwSize);
            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, raw, &dwSize, sizeof(RAWINPUTHEADER));

            if (raw->header.dwType == RIM_TYPEMOUSE) {
                g_inputData.mouseX += raw->data.mouse.lLastX;
                g_inputData.mouseY += raw->data.mouse.lLastY;
            }
            free(raw);
            break;
        }
        // Also handle standard Windows mouse-move messages so we get deltas
        // when the cursor is inside the client area even if no button is pressed.
        case WM_MOUSEMOVE: {
            int x = LOWORD(lParam);
            int y = HIWORD(lParam);
            // On the very first move we should initialize the last position
            // so we don't get a huge delta from an uninitialized value.
            static bool initialized = false;
            if (!initialized) {
                g_lastMouseX = x;
                g_lastMouseY = y;
                initialized = true;
            }
            g_inputData.mouseX += (x - g_lastMouseX);
            g_inputData.mouseY += (y - g_lastMouseY);
            g_lastMouseX = x;
            g_lastMouseY = y;
            break;
        }
        // Update mouse pressed flag and position on left button press.
        case WM_LBUTTONDOWN:
            g_inputData.mousePressed = TRUE;
            g_inputData.mouseX = LOWORD(lParam);
            g_inputData.mouseY = HIWORD(lParam);
            // Initialize last mouse pos on button down so subsequent WM_MOUSEMOVE
            // deltas are correct even if WM_MOUSEMOVE hasn't been seen yet.
            g_lastMouseX = g_inputData.mouseX;
            g_lastMouseY = g_inputData.mouseY;
            break;
        // Reset mouse pressed flag on left button release.
        case WM_LBUTTONUP:
            g_inputData.mousePressed = FALSE;
            break;
        // Mark key as pressed when a key is pressed.
        case WM_KEYDOWN:
            if (wParam < 256)
                g_inputData.keys[wParam] = TRUE;
            break;
        // Mark key as released when a key is released.
        case WM_KEYUP:
            if (wParam < 256)
                g_inputData.keys[wParam] = FALSE;
            break;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            PaintWindow(hdc);
            EndPaint(hwnd, &ps);
        } break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_ERASEBKGND:
            return 1; // Prevents background erasure to reduce flicker.
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

/////////////////////////////////////////////////////////////////////////This thing registers the window class (which has all the info about the window) with the OS
BOOL WinLib_Init(HINSTANCE hInstance) {
    // Fill in the WNDCLASSEX structure.
    WNDCLASSEX wc = {0};
    wc.cbSize        = sizeof(WNDCLASSEX);
    wc.style         = CS_HREDRAW | CS_VREDRAW;   // Redraw on horizontal/vertical size changes.
    wc.lpfnWndProc   = WinLib_WindowProc;         // Our window procedure.
    wc.cbClsExtra    = 0;
    wc.cbWndExtra    = 0;
    wc.hInstance     = hInstance;
    wc.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = NULL;  // Set a default background color.
    wc.lpszMenuName  = NULL;
    wc.lpszClassName = WINLIB_CLASS_NAME;
    wc.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);

    // Register the window class.
    return RegisterClassEx(&wc);
}
///////////////////////////////////////////////////////////////////////This thing creates the window with the registered class and other stuff it plugs in
WinWindow* WinLib_CreateWindow(const char* title, int width, int height, HINSTANCE hInstance) {
    // Allocate memory for our window structure.
    WinWindow* window = (WinWindow*)malloc(sizeof(WinWindow));
    if (!window)
        return NULL;

    // Create the window using the previously registered class.
    window->hwnd = CreateWindowEx(
        0,                      // Optional window styles.
        WINLIB_CLASS_NAME,      // Window class name.
        title,                  // Window title.
        WS_OVERLAPPEDWINDOW,    // Window style (includes title bar, border, etc.).
        CW_USEDEFAULT, CW_USEDEFAULT,  // Initial position (x, y).
        width, height,          // Window dimensions.
        NULL,                   // Parent window (none in this case).
        NULL,                   // Menu handle.
        hInstance,              // Application instance handle.
        NULL                    // Additional application data.
    );

    // If window creation failed, free allocated memory and return NULL.
    if (!window->hwnd) {
        free(window);
        return NULL;
    }

    // Make the window visible and update it.
    ShowWindow(window->hwnd, SW_SHOW);
    UpdateWindow(window->hwnd);
    // Register for raw mouse input so we receive relative mouse deltas.
    RAWINPUTDEVICE rid;
    rid.usUsagePage = 0x01; // Generic desktop controls
    rid.usUsage = 0x02;     // Mouse
    rid.dwFlags = 0;        // Deliver to this window when it has focus
    rid.hwndTarget = window->hwnd;
    RegisterRawInputDevices(&rid, 1, sizeof(rid));
    return window;
}

//////////////////////////////////////////////////////////////////////////EXTERMINATE a window
void WinLib_DestroyWindow(WinWindow* window) {
    if (window) {
        if (window->hwnd) {
            // Destroy the actual window.
            DestroyWindow(window->hwnd);
        }
        // Free the allocated memory.
        free(window);
    }
}
///////////////////////////////////////////////////////////////////////////////////This is for processing all of the pending events
bool WinLib_PollEvents(MSG* msg) {
    while (PeekMessage(msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg->message == WM_QUIT)
            return true; // Signal to exit
        TranslateMessage(msg);
        DispatchMessage(msg);
    }
    return false;
}
