#ifndef KEY_H
#define KEY_H

#include <stdint.h>
#include <X11/keysym.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KEY_MOD(code) (1 << (code))
#define KEY_ON(field, code) ((field) & KEY_MOD(code))
#define KEY_OFF(field, code) (!KEY_ON(field, code))

typedef enum {
        KEY_INPUT_W = 0, /* layout better for quick math ops */
        KEY_INPUT_S = 1,
        KEY_INPUT_A = 2,
        KEY_INPUT_D = 3,
        KEY_INPUT_ESC = 4,
        KEY_INPUT_SPACE = 5,
} key_code_t;

typedef uint32_t key_inputfield_t;

void key_setup(void);
void key_cleanup(void);
void key_poll_event(void);

key_inputfield_t key_get_pressed(void);
key_inputfield_t key_get_typestart(void);
key_inputfield_t key_get_keyrelease(void);

#ifdef __cplusplus
}
#endif

#endif
