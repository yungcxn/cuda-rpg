
#include "world.h"
#include "impl/worldmain.h"
#include "impl/worldhome.h"
#include "ecs.h"


#define X_WORLD(name_id, loadfunc, updatefunc) loadfunc,
static const world_loadfunc_t world_loadfunc_table[] = {
    X_WORLDLIST
};
#undef X_WORLD

#define X_WORLD(name_id, loadfunc, updatefunc) updatefunc,
static const world_updatefunc_t world_updatefunc_table[] = {
    X_WORLDLIST
};
#undef X_WORLD

static world_ctx_t* world_ctx;
static world_updatefunc_t world_ctx_updatefunc;

static inline world_ctx_t* _world_ctx_alloc() {
    world_ctx_t* w = (world_ctx_t*)malloc(sizeof(world_ctx_t));
    if (!w) THROW("Failed to allocate memory for world context");
    return w;
}

void world_setup() {
        world_ctx = _world_ctx_alloc();
        world_ctx->ecs_handle = ecs_handled_create();
        world_ctx_firstload(WORLD_DEFAULT_ID);
}

void world_cleanup() {
        ecs_handled_destroy(&(world_ctx->ecs_handle));
        free(world_ctx);
}

static inline void _world_destroy_devdata(world_map_devdata_t* tiles) {
        if (!tiles) THROW("No tilelayers to free");
        free(tiles->bg);
        free(tiles->main);
        free(tiles->on_main);
        free(tiles->fg);
        tiles->bg = 0;
        tiles->main = 0;
        tiles->on_main = 0;
        tiles->fg = 0;
        tiles->dim.width4 = 0;
        tiles->dim.height4 = 0;
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
static inline void _world_ctx_reset() {
        ecs_zero_out(&(world_ctx->ecs_handle));
        _world_destroy(&(world_ctx->world));
}

void world_ctx_firstload(world_id_t world_id) {
        world_loadfunc_table[world_id](&world_ctx);
        world_ctx_updatefunc = world_updatefunc_table[world_id];
}

void world_ctx_load(world_id_t world_id) {
        _world_ctx_reset();
        world_ctx_firstload(world_id);
}

void world_ctx_update(float32_t dt) {
        if (!world_ctx_updatefunc) THROW("No world update function set");
        world_ctx_updatefunc(world_ctx, dt);
        ecs_update(&(world_ctx->ecs_handle), dt);
}

extern inline world_map_devdata_t world_create_devdata_t(
        tileinfo_id4_t* hostbg, tileinfo_id4_t* hostmain, tileinfo_id4_t* hostonmain, tileinfo_id4_t* hostfg,
        uint32_t width4, uint32_t height4
) {
        world_map_devdata_t data;
        if (hostbg) {
                cudamem_alloc(&data.bg, width4 * height4 * sizeof(tileinfo_id4_t));
                cudamem_copy(data.bg, hostbg, width4 * height4 * sizeof(tileinfo_id4_t), 1);
        } else {
                data.bg = 0;
        }

        if (hostmain) {
                cudamem_alloc(&data.main, width4 * height4 * sizeof(tileinfo_id4_t));
                cudamem_copy(data.main, hostmain, width4 * height4 * sizeof(tileinfo_id4_t), 1);
        } else {
                data.main = 0;
        }

        if (hostonmain) {
                cudamem_alloc(&data.on_main, width4 * height4 * sizeof(tileinfo_id4_t));
                cudamem_copy(data.on_main, hostonmain, width4 * height4 * sizeof(tileinfo_id4_t), 1);
        } else {
                data.on_main = 0;
        }

        if (hostfg) {
                cudamem_alloc(&data.fg, width4 * height4 * sizeof(tileinfo_id4_t));
                cudamem_copy(data.fg, hostfg, width4 * height4 * sizeof(tileinfo_id4_t), 1);
        } else {
                data.fg = 0;
        }

        data.dim.width4 = width4;
        data.dim.height4 = height4;
        return data;
}

extern inline world_map_beams_t world_create_map_beams(vec2f32_t* host_tls, vec2f32_t* host_brs, uint32_t beam_count) {
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
