#ifndef RENDER_CUH
#define RENDER_CUH

#include "../headeronly/vec.h"
#include <X11/Xlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void render_setup(void);
void render_cleanup(void);
void render(void);

Display* render_get_xdisplay(void);
Window render_get_xwindow(void);

#ifdef __cplusplus
}
#endif

#endif