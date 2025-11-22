
#define _POSIX_C_SOURCE 199309L /* for clock_gettime */
#include <time.h>
#include <stdbool.h>

#include "key.h"
#include "headeronly/def.h"
#include "headeronly/debugterm.h" /* IWYU pragma: keep */
#include "world/world.h"
#include "render/vulkan.h"
#include "render/render.h"
#include "controller.h"


world_ctx_t* world_ctx;
#ifdef DEBUG
static double fps_smoothed = 0.0;
#endif

static float64_t _get_time_seconds(void) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (float64_t) ts.tv_sec + (float64_t) ts.tv_nsec / 1e9;
}

static inline void _update(float64_t dt) {
        key_inputfield_t pressed = key_get_pressed();
        key_inputfield_t released = key_get_keyrelease();
        key_inputfield_t typestart = key_get_typestart();
        controller_apply(world_ctx, pressed, typestart, released);
        world_ctx_update(world_ctx, dt);

#ifdef DEBUG
        const double fps = (dt > 0.0) ? (1.0 / dt) : 0.0;
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * fps;
        debugterm_print("FPS: %.2f\n"
                        "pressed:   %032b\n"
                        "typestart: %032b\n"
                        "released:  %032b\n"
                        "-------------------------------------------\n"
                        "p-state:   %064b\n"
                        "p-pos:     (%.2f, %.2f)\n",
                        fps_smoothed, pressed, typestart, released, world_ctx->player->state,
                        (float64_t) world_ctx->player->pos1.x, 
                        (float64_t) world_ctx->player->pos1.y);
#endif
}

int main(void) {
#ifdef DEBUG
        debugterm_init();
#endif

        tex_realrgba_t** framebuffer_ptr = vulkan_setup();
        render_data_setup();
        key_setup(); /* after render; needs window/display */
        world_ctx = world_ctx_create();

        float64_t last_time = _get_time_seconds();

        while (true) {
                float64_t now = _get_time_seconds();
                float64_t dt = now - last_time;
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
#ifdef DEBUG
        debugterm_end();
#endif
        return 0;
}