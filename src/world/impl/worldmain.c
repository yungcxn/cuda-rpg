#include <stdalign.h>

#include "worldmain.h"
#include "../../render/tileinfo.h"
#include "../../render/util/cudamem.h"

#define WORLDWIDTH 12   /* in tiles */
#define WORLDHEIGHT 16  /* in tiles */

#define GA TILEINFO_ID_GRASS_A
#define VO TILEINFO_VOID
static const alignas(tileinfo_id4_t) tileinfo_id_t devtiles_main_layer[WORLDWIDTH][WORLDHEIGHT] = {
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
        { GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA, GA },
};
#undef GA
#undef NO

#define BEAM_COUNT 2

static const vec2f32_t beam_tls[BEAM_COUNT] = { {0.0f, 0.0f}, {32.0f, 48.0f} };
static const vec2f32_t beam_brs[BEAM_COUNT] = { {16.0f, 16.0f}, {48.0f, 64.0f} };

void worldmain_load(world_ctx_t* restrict world_ctx) {
        world_ctx->world.devtiles = world_create_devdata_t(
                0,
                (tileinfo_id4_t*)devtiles_main_layer,
                0,
                0,
                WORLDWIDTH / 4,
                WORLDHEIGHT
        );

        world_ctx->world.beams = world_create_map_beams(
                (vec2f32_t*)beam_tls,
                (vec2f32_t*)beam_brs,
                BEAM_COUNT
        );

        return &world_ctx->world;
}

void worldmain_update(world_ctx_t* world_ctx, float32_t dt) {
        ecs_t* ecs_instance = world_ctx->ecs_handle.instance;
        /* TODO */
}
