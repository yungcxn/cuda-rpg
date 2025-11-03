
#define _POSIX_C_SOURCE 199309L /* for clock_gettime */
#include <time.h>
#include <stdbool.h>

#include "key.h"
#include "headeronly/def.h"
#include "world/world.h"
#include "render/vulkan.h"
#include "render/render.h"

world_ctx_t* world_ctx;

static float32_t _get_time_seconds(void) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (float32_t) ts.tv_sec + (float32_t) ts.tv_nsec / 1e9;
}

static inline void _update(float32_t dt) {
        key_inputfield_t pressed = key_get_pressed();
        world_ctx_update(world_ctx, pressed, dt);
}

int main(void) {
        tex_realrgba_t* framebuffer = vulkan_setup();
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

                render(
                        framebuffer,
                        world_ctx->world.devtiles,
                        world_ctx->ecs_handle.instance->shared,
                        world_ctx->player->shared, dt
                );

                last_time = _get_time_seconds();
        }

        world_ctx_destroy(world_ctx);
        key_cleanup();
        vulkan_cleanup();
        render_data_cleanup();
        return 0;
}