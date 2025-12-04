#ifndef WINLIB_H
#define WINLIB_H

#ifdef __cplusplus
extern "C" {
#endif


#include <windows.h>

void PaintWindow(HDC hdc);
// A simple structure to hold our window handle.
typedef struct WinWindow {
    HWND hwnd;
} WinWindow;

// 1. Declare the InputData structure.
typedef struct InputData {
    int mouseX;           // Current mouse X position.
    int mouseY;           // Current mouse Y position.
    BOOL mousePressed;    // Whether the left mouse button is pressed.
    BOOL keys[256];       // State for each virtual key.
} InputData;

// 2. Declare the function that returns the current input state.
InputData WinLib_GetInputs(void);

/**
 * Initializes the window library by registering a window class.
 *
 * @param hInstance The instance handle of the application.
 * @return TRUE if registration is successful, FALSE otherwise.
 */
BOOL WinLib_Init(HINSTANCE hInstance);

/**
 * Creates a window with the specified title and dimensions.
 *
 * @param title     The title text of the window.
 * @param width     The width of the window.
 * @param height    The height of the window.
 * @param hInstance The instance handle of the application.
 * @return Pointer to a WinWindow structure on success, NULL on failure.
 */
WinWindow* WinLib_CreateWindow(const char* title, int width, int height, HINSTANCE hInstance);

/**
 * Destroys the window and cleans up any allocated resources.
 *
 * @param window Pointer to the WinWindow structure to be destroyed.
 */
void WinLib_DestroyWindow(WinWindow* window);

/**
 * Polls and dispatches pending window messages.
 * This function should be called regularly in your main loop.
 */
bool WinLib_PollEvents(MSG* msg);

#ifdef __cplusplus
}
#endif

#endif // WINLIB_H