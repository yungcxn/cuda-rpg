#ifndef TILEINFO_H
#define TILEINFO_H

#include <stdint.h>
#include "../headeronly/def.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TILEINFO_ANIM_SENTINEL -1.0f

/*   (ids of animation belonging together MUST be consecutive and end with TILEINFO_ANIM_SENTINEL) */

#define TID(tid) _TILEINFO_ID_##tid

#define X_TILEINFO_LIST \
        X(TID(VOID), 0, 0) \
        X(TID(GRASS_A), 1, 0) \
        X(TID(GRASS_B), 2, 2) \
        X(TID(GRASS_C), 2, 1) \
        \
        X(TID(GRASS_PATH_V), 3, 0) \
        X(TID(GRASS_PATH_H), 3, 1) \
        X(TID(GRASS_PATH_VTE), 3, 2) \
        X(TID(GRASS_PATH_HRE), 3, 3) \
        X(TID(GRASS_PATH_VBE), 3, 4) \
        X(TID(GRASS_PATH_HLE), 3, 5) \
        X(TID(GRASS_PATH_D1), 4, 0) \
        X(TID(GRASS_PATH_D2), 4, 1) \
        X(TID(GRASS_PATH_D3), 4, 2) \
        X(TID(GRASS_PATH_D4), 4, 3) \
        \
        X(TID(GRASS_SAND_TL), 5, 0) \
        X(TID(GRASS_SAND_L), 5, 1) \
        X(TID(GRASS_SAND_BL), 5, 2) \
        X(TID(GRASS_SAND_T), 6, 0) \
        X(TID(GRASS_SAND), 6, 1) \
        X(TID(GRASS_SAND_B), 6, 2) \
        X(TID(GRASS_SAND_TR), 7, 0) \
        X(TID(GRASS_SAND_R), 7, 1) \
        X(TID(GRASS_SAND_BR), 7, 2) \
        \
        XA(TID(WATER_0), 10, 1, 0.3f) \
        XA(TID(WATER_1), 10, 2, 0.3f) \
        XA(TID(WATER_2), 10, 3, 0.3f) \
        XAS(TID(WATER_3), 3) \
        \
        X(TID(GRASS_WATER_TL), 9, 3) \
        X(TID(GRASS_WATER_L), 9, 4) \
        X(TID(GRASS_WATER_BL), 9, 5) \
        X(TID(GRASS_WATER_T), 10, 0) \
        X(TID(GRASS_WATER_B), 10, 5) \
        X(TID(GRASS_WATER_TR), 11, 3) \
        X(TID(GRASS_WATER_R), 11, 4) \
        X(TID(GRASS_WATER_BR), 11, 5) \
        \
        XA(TID(FLOWER_T_0), 0, 11, 0.5f) \
        XA(TID(FLOWER_T_1), 1, 11, 0.5f) \
        XA(TID(FLOWER_T_2), 2, 11, 0.5f) \
        XAS(TID(FLOWER_T_3), 3) \
        XA(TID(FLOWER_B_0), 0, 12, 0.5f) \
        XA(TID(FLOWER_B_1), 1, 12, 0.5f) \
        XA(TID(FLOWER_B_2), 2, 12, 0.5f) \
        XAS(TID(FLOWER_B_3), 3) \
        \
        XA(TID(DUSTBALL_0), 3, 11, 0.2f) \
        XA(TID(DUSTBALL_1), 3, 12, 0.2f) \
        XA(TID(DUSTBALL_2), 3, 13, 0.2f) \
        XAS(TID(DUSTBALL_3), 3) \
        \
        X(TID(SINGLETREE_A), 4, 11) \
        X(TID(SINGLETREE_B), 5, 11) \
        X(TID(SINGLETREE_C), 4, 12) \
        X(TID(SINGLETREE_D), 5, 12) \
        X(TID(SINGLETREE_E), 4, 13) \
        X(TID(SINGLETREE_F), 5, 13) \
        X(TID(SINGLETREE_G), 4, 14) \
        X(TID(SINGLETREE_H), 5, 14) \
        \
        X(TID(PACKTREE1_A), 6, 11) \
        X(TID(PACKTREE1_B), 7, 11) \
        X(TID(PACKTREE1_C), 6, 12) \
        X(TID(PACKTREE1_D), 7, 12) \
        X(TID(PACKTREE1_E), 6, 13) \
        X(TID(PACKTREE1_F), 7, 13) \
        X(TID(PACKTREE1_G), 6, 14) \
        X(TID(PACKTREE1_H), 7, 14) \
        \
        X(TID(PACKTREE2_A), 8, 11) \
        X(TID(PACKTREE2_B), 9, 11) \
        X(TID(PACKTREE2_C), 8, 12) \
        X(TID(PACKTREE2_D), 9, 12)

/* the tileinfo_id_t corresponds to the index in the tileinfo table */
typedef uint16_t tileinfo_id_t;

typedef struct __attribute__((packed)) {
        uint16_t tx;
        uint16_t ty;
} tileinfo_t;

#define X(id, tx, ty) id,
#define XA(id, tx, ty, anim_speed) id,
#define XAS(id, animlen) id,
enum {
        X_TILEINFO_LIST
        TILEINFOS
}; /* tileinfo_id_t */
#undef X
#undef XA
#undef XAS

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
