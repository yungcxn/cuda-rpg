#include "worldmain.h"


static inline world_map_tiles_t* _worldamain_init_map_tiles() {}

static inline world_map_borders_t* _worldmain_init_map_borders() {}

world_t* worldmain_load(ecs_handle_t* ecs_handle) {
        world_t* world = (world_t*)malloc(sizeof(world_t));
        if (!world) THROW("Failed to allocate memory for world");
        world->tiles = _worldamain_init_map_tiles();
        world->borders = _worldmain_init_map_borders();
        return world;
} 

void worldmain_update(world_ctx_t* world_ctx, float32_t dt) {
        ecs_t* ecs_instance = world_ctx->ecs_handle.instance;
        /* TODO */
}
