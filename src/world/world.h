#ifndef WORLD_H
#define WORLD_H

#include <stdint.h>
#include "../def.h"
#include "../types/vec.h"
#include "../render/tileinfo.h"
#include "ecs.h"


#define X_WORLDLIST \
/*      X_WORLD(name_id, loadfunc, updatefunc) */ \
        X_WORLD(WORLDMAIN, worldmain_load, worldmain_update) \
        X_WORLD(WORLDHOME, worldhome_load, worldhome_update)

#define X_WORLD(name_id, loadfunc, updatefunc) name_id,
typedef enum {
        X_WORLDLIST
        WORLD_COUNT
} world_id_t;
#undef X_WORLD

#define WORLD_DEFAULT_ID 0

typedef vec2u32_t world_border_t;

typedef struct {
        tileinfo_id4_t* tiles; /* size is 64bit */
        uint32_t width4;
        uint32_t height4;
} world_map_tilelayer_t;

typedef struct __attribute__((packed)) {
        world_map_tilelayer_t* bg;
        world_map_tilelayer_t* main;
        world_map_tilelayer_t* on_main;
        world_map_tilelayer_t* on_lense;
} world_map_tiles_t;

typedef struct {
        world_border_t* borders; /* size is 64bit */
        uint32_t border_count;
} world_map_borders_t;

typedef struct {
        world_map_tiles_t* tiles;
        world_map_borders_t* borders;
} world_t; /* note: these need to dynamically loaded   */
           /*       so no full table OF ALL SUBFIELDS! */

typedef struct {
        world_t* world;
        ecs_handle_t ecs_handle;
        /* event handling ?! TODO */
} world_ctx_t; /* current world context */

typedef world_t* (*world_loadfunc_t)(ecs_handle_t* ecs_handle);
typedef void (*world_updatefunc_t)(world_ctx_t* world_ctx, float32_t dt);

void world_setup();
void world_cleanup();
void world_ctx_update(float dt);
void world_ctx_load(world_id_t world_id); /* unload func is prv*/

#endif