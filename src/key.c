#include "key.h"
#include <X11/keysym.h>
#include <X11/Xlib.h>
#include "render/vulkan.h"
#include "headeronly/def.h"

static Display* x_display = 0;
static Window   x_window  = 0;

static key_inputfield_t current_presseds = 0;
static key_inputfield_t current_typestarts = 0;
static key_inputfield_t current_keyreleases = 0;
static key_inputfield_t _pressed_before = 0;

void key_setup() {
        x_display = vulkan_get_xdisplay();
        x_window = vulkan_get_xwindow();
        if (!x_display || !x_window)
                THROW("Failed to get X11 Display or Window from renderer");
        XSelectInput(x_display, x_window, KeyPressMask | KeyReleaseMask);
}

void key_cleanup(void) {
        x_display = 0;
        x_window = 0;
}

void key_poll_event(void) {
        if (!x_display) THROW("X11 Display not initialized in key module");
        
        _pressed_before = current_presseds;
        current_typestarts = 0;
        current_keyreleases = 0;
        XEvent ev;
        while (XPending(x_display)) {
                XNextEvent(x_display, &ev);
                if (ev.type != KeyPress && ev.type != KeyRelease)
                        continue;
                KeySym sym = XLookupKeysym(&ev.xkey, 0);
                uint32_t k;
                switch (sym) {
                        case XK_w:
                        case XK_W:      k = KEY_INPUT_W; break;
                        case XK_s:
                        case XK_S:      k = KEY_INPUT_S; break;
                        case XK_a:
                        case XK_A:      k = KEY_INPUT_A; break;
                        case XK_d:
                        case XK_D:      k = KEY_INPUT_D; break;
                        case XK_Escape: k = KEY_INPUT_ESC; break;
                        case XK_space:  k = KEY_INPUT_SPACE; break;
                        default: continue;
                }
                if (ev.type == KeyPress) {
                        current_presseds |= KEY_MOD(k);
                        if (!KEY_ON(_pressed_before, k))
                                current_typestarts |= KEY_MOD(k);
                } else {
                        if (XPending(x_display)) {
                                XEvent next;
                                XPeekEvent(x_display, &next);
                                if (next.type == KeyPress &&
                                    next.xkey.time == ev.xkey.time &&
                                    next.xkey.keycode == ev.xkey.keycode)
                                        continue;
                        }
                        current_presseds &= ~KEY_MOD(k);
                        current_keyreleases |= KEY_MOD(k);
                }
        }
}


key_inputfield_t key_get_pressed(void) {
        return current_presseds;
}

key_inputfield_t key_get_typestart(void) {
        return current_typestarts;
}

key_inputfield_t key_get_keyrelease(void) {
        return current_keyreleases;
}