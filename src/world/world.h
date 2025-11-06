#ifndef WORLD_H
#define WORLD_H

#include <stdint.h>
#include "../headeronly/def.h"
#include "../headeronly/vec.h"
#include "../render/tileinfo.h"
#include "ecs.h"
#include "player/player.h"


#ifdef __cplusplus
extern "C" {
#endif

#define WORLD_DEVTILELAYERS (sizeof((world_map_devtiles_t){0}) / sizeof(world_map_devtilelayer_t*))

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

typedef struct {
        tileinfo_id_t* bg;             /* dyn-alloc and destroy per load */
        tileinfo_id_t* main;           /* dyn-alloc and destroy per load */
        tileinfo_id_t* on_main;        /* dyn-alloc and destroy per load */
        tileinfo_id_t* fg;             /* dyn-alloc and destroy per load */
        struct {
                uint32_t width;
                uint32_t height;
        } dim;
} world_map_devdata_t;

#define WORLD_MAP_LAYERS 4

typedef struct {
        vec2f32_t* beam_tls;            /* dyn-alloc and destroy per load */
        vec2f32_t* beam_brs;            /* dyn-alloc and destroy per load */
        uint32_t beam_count;
} world_map_beams_t;

typedef struct {
        world_map_devdata_t devtiles;
        world_map_beams_t beams;
} world_t; /* note: these need to dynamically loaded   */
           /*       so no full table OF ALL SUBFIELDS! */

typedef struct {
        world_t world;
        ecs_handle_t ecs_handle;
        player_t* player;
        /* event handling ?! TODO */
} world_ctx_t; /* current world context */

typedef void (*world_loadfunc_t)(world_ctx_t* world_ctx);
typedef void (*world_updatefunc_t)(world_ctx_t* world_ctx, float32_t dt);

world_ctx_t* world_ctx_create();
void world_ctx_destroy(world_ctx_t* world_ctx);
void world_ctx_update(world_ctx_t* world_ctx, key_inputfield_t pressed_keys, float32_t dt);
void world_ctx_load(world_ctx_t* world_ctx, world_id_t world_id); /* unload func is prv */

/* for impl to use: */
world_map_devdata_t world_create_devdata_t(tileinfo_id_t* hostbg, tileinfo_id_t* hostmain,
                                           tileinfo_id_t* hostonmain, tileinfo_id_t* hostfg,
                                           uint32_t width, uint32_t height);

world_map_beams_t world_create_map_beams(vec2f32_t* host_tls, vec2f32_t* host_brs,
                                         uint32_t beam_count);

#ifdef __cplusplus
}
#endif

#endif
