
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

static inline void _world_free_tilelayer(world_map_tilelayer_t* layer) {
        if (layer) {
                if (layer->tiles) free(layer->tiles);
                free(layer);
        }
}

static inline void _world_free_tilelayers(world_map_tiles_t* tiles) {
        if (tiles) {
                _world_free_tilelayer(tiles->bg);
                _world_free_tilelayer(tiles->main);
                _world_free_tilelayer(tiles->on_main);
                _world_free_tilelayer(tiles->on_lense);
                free(tiles);
        }
}

static inline void _world_free_borders(world_map_borders_t* borders) {
        if (borders) {
                free(borders->borders);
                free(borders);
        }
}

static inline void _world_freeall(world_t* world) {
        if (world) {
                _world_free_tilelayers(world->tiles);
                _world_free_borders(world->borders);
                free(world);
        }
}

/* destroy all world points to and what that points to aso... but zero out ecs */
static inline void _world_ctx_reset() {
        ecs_zero_out(&(world_ctx->ecs_handle));
        _world_freeall(world_ctx->world);
        world_ctx->world = 0;
}

void world_ctx_firstload(world_id_t world_id) {
        world_ctx = world_loadfunc_table[world_id]();
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