
#include "world.h"
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

static world_updatefunc_t world_ctx_updatefunc;

static ecs_t* ecs_instance;
static world_ctx_t* world_ctx;

void world_setup() {
        ecs_setup();
        ecs_instance = ecs_get();
}

void world_cleanup() {
        ecs_cleanup();
}

void world_ctx_update(float32_t dt) {
        ecs_update(dt);
        world_ctx_updatefunc(dt);
}

void world_ctx_load() {
        /* TODO: ecs cleanup */
        if (world_ctx_loadfunc) free(world_ctx);
        world_ctx = world_loadfunc_table[WORLDMAIN]();
        world_ctx_updatefunc = world_updatefunc_table[WORLDMAIN];
}

void world_ctx_update(float32_t dt) {
        if (!world_ctx_updatefunc) THROW("No world update function set");
        world_ctx_updatefunc(dt);
}