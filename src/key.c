
#include "key.h"

#include <X11/keysym.h>
#include <X11/Xlib.h>
#include <stdlib.h>
#include <string.h>

#include "render/render.h"
#include "def.h"

const uint32_t key_x11_to_code_table[256] = { /* since the XKeys go usually up to 0xFF */
        [XK_W] = KEY_INPUT_W,
        [XK_w] = KEY_INPUT_W,
        [XK_A] = KEY_INPUT_A,
        [XK_a] = KEY_INPUT_A,
        [XK_S] = KEY_INPUT_S,
        [XK_s] = KEY_INPUT_S,
        [XK_D] = KEY_INPUT_D,
        [XK_d] = KEY_INPUT_D,
        [XK_Escape & 0xFF] = KEY_INPUT_ESC, /* hack; does not clash with other common key */
        [XK_space] = KEY_INPUT_SPACE,
};

static Display* x_display = 0;
static Window   x_window  = 0;

static key_inputfield_t current_presseds = 0;
static key_inputfield_t current_typestarts = 0; 
static key_inputfield_t _pressed_before = 0;

void key_setup() {
        x_display = render_get_xdisplay();
        x_window  = render_get_xwindow();
        if (!x_display || !x_window) THROW("Failed to get X11 Display or Window from renderer");

        /* ensure the window is selecting key events */
        XSelectInput(x_display, x_window, KeyPressMask | KeyReleaseMask);
}

void key_cleanup(void) {
        /* nothing to free here, we don't own the Display/Window */
        x_display = 0;
        x_window = 0;
}

void key_poll_event(void) {
        if (!x_display) THROW("X11 Display not initialized in key module");
        if (XPending(x_display) == 0) return;

        /* we must reset typestarts, since we do not carry over per frame */
        _pressed_before = current_presseds;
        XEvent ev;
        XNextEvent(x_display, &ev);
        if (ev.type == KeyPress) {
                KeyCode keycode = ev.xkey.keycode & 0xFF;
                uint32_t key = key_x11_to_code_table[keycode]; 
                current_presseds |= KEY_MOD(key);

                if (KEY_ON(_pressed_before, key) && KEY_ON(current_presseds, key)) {
                        current_typestarts &= ~KEY_MOD(key);
                } else {
                        current_typestarts |= KEY_MOD(key);
                }
        } else if (ev.type == KeyRelease) {
                /* we do not remove from typestarts, gets cleared next frame, but only presseds */
                KeyCode keycode = ev.xkey.keycode & 0xFF;
                uint32_t key = key_x11_to_code_table[keycode];
                current_presseds &= ~KEY_MOD(key);
        }
}

key_inputfield_t key_get_pressed(void) {
        return current_presseds;
}

key_inputfield_t key_get_typestart(void) {
        return current_typestarts;
}