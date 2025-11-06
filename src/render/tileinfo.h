#ifndef TILEINFO_H
#define TILEINFO_H

#include <stdint.h>
#include "../headeronly/def.h"
#include "../headeronly/tileinfogrid.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TILEINFO_ANIM_SENTINEL -1.0f

/* X_TILEINFO_TUPLE(tileinfo_id, tx, ty) */
/* X_TILEINFO_TUPLE_ANIM(tileinfo_id, tx, ty, framelen_in_seconds) */
/*   (ids of animation belonging together MUST be consecutive and end with TILEINFO_ANIM_SENTINEL) */
/* X_TILEINFO_GRIDTUPLE(id, tx, ty, A, B) */
/* X_TILEINFO_GRIDTUPLE_ANIM(id, tx, ty, A, B, frame) */

#define X_TILEINFO_TUPLE_ANIM_SENTINEL(tileinfo_id, animlen) \
        X_TILEINFO_TUPLE_ANIM(tileinfo_id, 0, 0, (float32_t) -animlen)

#define TILEINFO_LIST \
        X_TILEINFO_TUPLE(TILEINFO_ID_VOID, 0, 0) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_A, 1, 0) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_B, 2, 2) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_C, 2, 1) \
        \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_PATH_V, 3, 0) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_PATH_H, 3, 1) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_PATH_VTE, 3, 2) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_PATH_HRE, 3, 3) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_PATH_VBE, 3, 4) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_PATH_HLE, 3, 5) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_PATH_D1, 4, 0) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_PATH_D2, 4, 1) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_PATH_D3, 4, 2) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_PATH_D4, 4, 3) \
        \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_SAND_TL, 5, 0) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_SAND_L, 5, 1) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_SAND_BL, 5, 2) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_SAND_T, 6, 0) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_SAND, 6, 1) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_SAND_B, 6, 2) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_SAND_TR, 7, 0) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_SAND_R, 7, 1) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_SAND_BR, 7, 2) \
        \
        X_TILEINFO_TUPLE_ANIM(TILEINFO_ID_WATER_0, 10, 1, 0.3f) \
        X_TILEINFO_TUPLE_ANIM(TILEINFO_ID_WATER_1, 10, 2, 0.3f) \
        X_TILEINFO_TUPLE_ANIM(TILEINFO_ID_WATER_2, 10, 3, 0.3f) \
        X_TILEINFO_TUPLE_ANIM_SENTINEL(TILEINFO_ID_WATER_3, 3) \
        \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_WATER_TL, 9, 3) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_WATER_L, 9, 4) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_WATER_BL, 9, 5) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_WATER_T, 10, 0) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_WATER_B, 10, 5) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_WATER_TR, 11, 3) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_WATER_R, 11, 4) \
        X_TILEINFO_TUPLE(TILEINFO_ID_GRASS_WATER_BR, 11, 5) \
        \
        X_TILEINFO_GRIDTUPLE_ANIM(TILEINFO_ID_FLOWER_0, 0, 11, 1, 2, 0.5f) \
        X_TILEINFO_GRIDTUPLE_ANIM(TILEINFO_ID_FLOWER_1, 1, 11, 1, 2, 0.5f) \
        X_TILEINFO_GRIDTUPLE_ANIM(TILEINFO_ID_FLOWER_2, 2, 11, 1, 2, 0.5f) \
        X_TILEINFO_TUPLE_ANIM_SENTINEL(TILEINFO_ID_FLOWER_3, 3) \
        \
        X_TILEINFO_TUPLE_ANIM(TILEINFO_ID_DUSTBALL_0, 3, 11, 0.2f) \
        X_TILEINFO_TUPLE_ANIM(TILEINFO_ID_DUSTBALL_1, 3, 12, 0.2f) \
        X_TILEINFO_TUPLE_ANIM(TILEINFO_ID_DUSTBALL_2, 3, 13, 0.2f) \
        X_TILEINFO_TUPLE_ANIM_SENTINEL(TILEINFO_ID_DUSTBALL_3, 3) \
        \
        X_TILEINFO_GRIDTUPLE(TILEINFO_ID_TREE, 4, 11, 2, 4) \
        X_TILEINFO_GRIDTUPLE(TILEINFO_ID_TREEPACK_A, 6, 11, 2, 4) \
        X_TILEINFO_GRIDTUPLE(TILEINFO_ID_TREEPACK_B, 8, 11, 2, 2) 

/* the tileinfo_id_t corresponds to the index in the tileinfo table */
typedef uint16_t tileinfo_id_t;

typedef struct __attribute__((packed)) {
        uint16_t tx;
        uint16_t ty;
} tileinfo_t;

#define X_TILEINFO_TUPLE(id, tx, ty) id,
#define X_TILEINFO_TUPLE_ANIM(id, tx, ty, anim_speed) id,
enum {
        TILEINFO_LIST
        TILEINFOS /* not beautiful, but little hack since __countof is unsupported in nvcc */
}; /* tileinfo_id_t */
#undef X_TILEINFO_TUPLE
#undef X_TILEINFO_TUPLE_ANIM


#define TILEINFO(tx, ty) {tx, ty}

tileinfo_t* tileinfo_devtable_create(void);
void tileinfo_devtable_destroy(tileinfo_t* tileinfo_devtable);
float32_t* tileinfo_animlen_devtable_create(void);
void tileinfo_animlen_devtable_destroy(float32_t* tileinfo_animlen_devtable);
float32_t* tileinfo_animtimer_devcache_create(void);
void tileinfo_animtimer_devcache_destroy(float32_t* tileinfo_animtimer_devcache);

#ifdef __cplusplus
}
#endif

#endif
