#include <stdalign.h>

#include "worldmain.h"
#include "../../render/tileinfo.h"

#define WORLDMAIN_WIDTH 20   /* in tiles */
#define WORLDMAIN_HEIGHT 16  /* in tiles */

#define GA TID(GRASS_A)
#define VO TID(VOID)
static const tileinfo_id_t devtiles_main_layer[WORLDMAIN_HEIGHT][WORLDMAIN_WIDTH] = {
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
};
#undef GA
#undef NO

#define BEAM_COUNT 2

static const vec2f32_t beam_tls[BEAM_COUNT] = { {0.0f, 0.0f}, {32.0f, 48.0f} };
static const vec2f32_t beam_brs[BEAM_COUNT] = { {16.0f, 16.0f}, {48.0f, 64.0f} };

void worldmain_load(world_ctx_t* world_ctx) {
        world_ctx->world.devtiles = world_create_devdata_t(0, (tileinfo_id_t*)devtiles_main_layer,
                                                           0, 0, WORLDMAIN_WIDTH, WORLDMAIN_HEIGHT);

        world_ctx->world.beams = world_create_map_beams((vec2f32_t*)beam_tls, (vec2f32_t*)beam_brs,
                                                        BEAM_COUNT);
}

void worldmain_update(world_ctx_t* world_ctx, float32_t dt) {
        ecs_t* ecs_instance = world_ctx->ecs_handle.instance;
        /* TODO */
}
