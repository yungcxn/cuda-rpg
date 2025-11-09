
#define _POSIX_C_SOURCE 199309L /* for clock_gettime */
#include <time.h>
#include <stdbool.h>

#include "key.h"
#include "headeronly/def.h"
#include "world/world.h"
#include "render/vulkan.h"
#include "render/render.h"
#include "controller.h"

world_ctx_t* world_ctx;

static float32_t _get_time_seconds(void) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (float32_t) ts.tv_sec + (float32_t) ts.tv_nsec / 1e9;
}

static inline void _update(float32_t dt) {
        key_inputfield_t pressed = key_get_pressed();
        key_inputfield_t released = key_get_keyrelease();
        controller_apply(world_ctx, pressed, released);
        world_ctx_update(world_ctx, dt);
}

int main(void) {
        tex_realrgba_t** framebuffer_ptr = vulkan_setup();
        render_data_setup();
        key_setup(); /* after render; needs window/display */
        world_ctx = world_ctx_create();

        float32_t last_time = _get_time_seconds();

        while (true) {
                float32_t now = _get_time_seconds();
                float32_t dt = now - last_time;
                last_time = now;

                key_poll_event();

                _update(dt);

                render(framebuffer_ptr, world_ctx->world.devtiles, world_ctx->ecs_handle,
                       world_ctx->player, dt);
        }

        world_ctx_destroy(world_ctx);
        key_cleanup();
        vulkan_cleanup();
        render_data_cleanup();
        return 0;
}