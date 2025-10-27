
#define _POSIX_C_SOURCE 199309L /* for clock_gettime */
#include <time.h>

#include "key.h"
#include "def.h"
#include "types/vec.h"

#include "world/world.h"

#include "render/render.h"



static float32_t _get_time_seconds(void) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (float32_t) ts.tv_sec + (float32_t) ts.tv_nsec / 1e9;
}

static inline void _update(float32_t dt) {
        key_inputfield_t pressed = key_get_pressed();
        float32_t camspeed = 10.0f * dt; /* 10 pixels per second */
        /* 
        if (KEY_ON(pressed, KEY_INPUT_W)) render_offset_cam(VEC2F(0.0f, -camspeed));
        if (KEY_ON(pressed, KEY_INPUT_S)) render_offset_cam(VEC2F(0.0f, camspeed));
        if (KEY_ON(pressed, KEY_INPUT_A)) render_offset_cam(VEC2F(-camspeed, 0.0f));
        if (KEY_ON(pressed, KEY_INPUT_D)) render_offset_cam(VEC2F(camspeed, 0.0f));
        */
        world_ctx_update(dt);
}


int main(void) {

        render_setup();
        key_setup(); /* after render; needs window/display */
        world_setup();

        float32_t last_time = _get_time_seconds();

        while (true) {
                float32_t now = _get_time_seconds();
                float32_t dt = now - last_time;
                last_time = now;

                key_poll_event();
                _update(dt);
                render();
                last_time = _get_time_seconds();
        }

        world_cleanup();
        key_cleanup();
        render_cleanup();
        return 0;
}