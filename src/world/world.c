#include <string.h>

#include "world.h"
#include "impl/worldmain.h" /* IWYU pragma: keep for X */
#include "impl/worldhome.h" /* IWYU pragma: keep for X_WORXLD_TUPLE */
#include "ecs.h"
#include "../render/util/ccuda.h"


#define X(name_id, loadfunc, updatefunc) loadfunc,
static const world_loadfunc_t world_loadfunc_table[] = {
    X_WORLDLIST
};
#undef X

#define X(name_id, loadfunc, updatefunc) updatefunc,
static const world_updatefunc_t world_updatefunc_table[] = {
    X_WORLDLIST
};
#undef X

static world_updatefunc_t world_ctx_updatefunc;

void world_ctx_destroy(world_ctx_t* world_ctx) {
        ecs_handled_destroy(&(world_ctx->ecs_handle));
        player_destroy(world_ctx->player);
        free(world_ctx);
}

static inline void _world_destroy_devdata(world_map_devdata_t* tiles) {
        if (!tiles) THROW("No tilelayers to free");
        ccuda_free(tiles->bg);
        ccuda_free(tiles->main);
        ccuda_free(tiles->on_main);
        ccuda_free(tiles->fg);
        tiles->bg = 0;
        tiles->main = 0;
        tiles->on_main = 0;
        tiles->fg = 0;
        tiles->dim.width = 0;
        tiles->dim.height = 0;
}

static inline void _world_destroy_beam_points(world_map_beams_t* beams) {
        if (!beams) THROW("No beamdata to free");
        free(beams->beam_tls);
        free(beams->beam_brs);
        beams->beam_tls = 0;
        beams->beam_brs = 0;
        beams->beam_count = 0;
}

static inline void _world_destroy(world_t* world) {
        if (!world) THROW("No world to destroy");
        _world_destroy_devdata(&(world->devtiles));
        _world_destroy_beam_points(&(world->beams));
}

/* destroy all world points to and what that points to aso... but zero out ecs */
static inline void _world_ctx_reset(world_ctx_t* world_ctx) {
        ecs_zero_out(&(world_ctx->ecs_handle));
        _world_destroy(&(world_ctx->world));
}

void world_ctx_firstload(world_ctx_t* world_ctx, world_id_t world_id) {
        world_loadfunc_table[world_id](world_ctx);
        world_ctx_updatefunc = world_updatefunc_table[world_id];
}

void world_ctx_load(world_ctx_t* world_ctx, world_id_t world_id) {
        _world_ctx_reset(world_ctx);
        world_ctx_firstload(world_ctx, world_id);
}

void world_ctx_update(world_ctx_t* world_ctx, float32_t dt) {
        if (!world_ctx_updatefunc) THROW("No world update function set");

        world_ctx_updatefunc(world_ctx, dt);

        player_update(world_ctx->player, dt);

        ecs_update(&(world_ctx->ecs_handle), dt);
}

world_ctx_t* world_ctx_create() {
        uint32_t size = sizeof(world_ctx_t);
        uint32_t aligned_size = (size + 31) & ~31;
        world_ctx_t* w = (world_ctx_t*)aligned_alloc(32, aligned_size);
        if (!w) THROW("Failed to allocate memory for world context");
        w->ecs_handle = ecs_handled_create();
        w->player = player_create();
        world_ctx_firstload(w, WORLD_DEFAULT_ID);
        return w;
}

world_map_devdata_t world_create_devdata_t(
        tileinfo_id_t* hostbg, tileinfo_id_t* hostmain, tileinfo_id_t* hostonmain, 
        tileinfo_id_t* hostfg, uint32_t width, uint32_t height
) {
        world_map_devdata_t data;
        if (hostbg) {
                ccuda_malloc((void**)&data.bg, width * height * sizeof(tileinfo_id_t));
                ccuda_copy(data.bg, hostbg, width * height * sizeof(tileinfo_id_t), 1);
        } else {
                data.bg = 0;
        }

        if (hostmain) {
                ccuda_malloc((void**)&data.main, width * height * sizeof(tileinfo_id_t));
                ccuda_copy(data.main, hostmain, width * height * sizeof(tileinfo_id_t), 1);
        } else {
                data.main = 0;
        }

        if (hostonmain) {
                ccuda_malloc((void**)&data.on_main, width * height * sizeof(tileinfo_id_t));
                ccuda_copy(data.on_main, hostonmain, width * height * sizeof(tileinfo_id_t), 1);
        } else {
                data.on_main = 0;
        }

        if (hostfg) {
                ccuda_malloc((void**)&data.fg, width * height * sizeof(tileinfo_id_t));
                ccuda_copy(data.fg, hostfg, width * height * sizeof(tileinfo_id_t), 1);
        } else {
                data.fg = 0;
        }

        data.dim.width = width;
        data.dim.height = height;
        return data;
}

world_map_beams_t world_create_map_beams(vec2f32_t* host_tls, vec2f32_t* host_brs, 
                                         uint32_t beam_count) {

        if (!host_tls || !host_brs) THROW("No beam data provided");
        world_map_beams_t beams;
        beams.beam_tls = (vec2f32_t*)malloc(beam_count * sizeof(vec2f32_t));
        if (!beams.beam_tls) THROW("Failed to allocate memory for beam tls");
        beams.beam_brs = (vec2f32_t*)malloc(beam_count * sizeof(vec2f32_t));
        if (!beams.beam_brs) THROW("Failed to allocate memory for beam brs");
        memcpy(beams.beam_tls, host_tls, beam_count * sizeof(vec2f32_t));
        memcpy(beams.beam_brs, host_brs, beam_count * sizeof(vec2f32_t));
        beams.beam_count = beam_count;
        return beams;
}
