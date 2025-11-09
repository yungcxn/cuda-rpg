#ifndef SPRITEINFO_H
#define SPRITEINFO_H

#include "../headeronly/vec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPRITEINFO_ANIM_NOANIMDURATION 0.0f

#define X_SPRITEINFO_TUPLE_ANIM_SENTINEL(spriteinfo_id, animlen) \
        X_SPRITEINFO_TUPLE_ANIM(spriteinfo_id, 0, 0, 0, 0, 0, 0, (float32_t) -animlen)

#define SID(sid) _SPRITEINFO_ID_##sid

#define X_SPRITEINFO_LIST \
        X(SID(NONE), 0, 0, 0, 0, 0, 0) \
        X(SID(PRINCESS_IDLE_D), 0, 33, 1, 2, 0, -1) \
        X(SID(PRINCESS_IDLE_U), 1, 33, 1, 2, 0, -1) \
        X(SID(PRINCESS_IDLE_R), 2, 33, 1, 2, 0, -1) \
        X(SID(PRINCESS_IDLE_L), 3, 33, 1, 2, 0, -1) \
        \
        XA(SID(PLAYER_IDLE_D_0), 0, 35, 1, 2, 0, -1, 1.0f) \
        XA(SID(PLAYER_IDLE_D_1), 1, 35, 1, 2, 0, -1, 1.0f) \
        XAS(SID(PLAYER_IDLE_D_2), 2) \
        \
        X(SID(PLAYER_IDLE_U), 2, 35, 1, 2, 0, 0) \
        X(SID(PLAYER_IDLE_R), 3, 35, 1, 2, 0, 0) \
        X(SID(PLAYER_IDLE_L), 4, 35, 1, 2, 0, 0) \
        \
        XA(SID(PLAYER_RUN_D_0), 0, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_D_1), 1, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_D_2), 2, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_D_3), 3, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_D_4), 4, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_D_5), 5, 37, 1, 2, 0, -1, 0.1f) \
        XAS(SID(PLAYER_RUN_D_6), 6) \
        \
        XA(SID(PLAYER_RUN_U_0), 6, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_U_1), 7, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_U_2), 8, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_U_3), 9, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_U_4), 10, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_U_5), 11, 37, 1, 2, 0, -1, 0.1f) \
        XAS(SID(PLAYER_RUN_U_6), 6) \
        \
        XA(SID(PLAYER_RUN_R_0), 12, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_R_1), 13, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_R_2), 14, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_R_3), 15, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_R_4), 16, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_R_5), 17, 37, 1, 2, 0, -1, 0.1f) \
        XAS(SID(PLAYER_RUN_R_6), 6) \
        \
        XA(SID(PLAYER_RUN_L_0), 18, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_L_1), 19, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_L_2), 20, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_L_3), 21, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_L_4), 22, 37, 1, 2, 0, -1, 0.1f) \
        XA(SID(PLAYER_RUN_L_5), 23, 37, 1, 2, 0, -1, 0.1f) \
        XAS(SID(PLAYER_RUN_L_6), 6) \
        \
        XA(SID(PLAYER_ROLL_U_0), 0, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_U_1), 1, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_U_2), 2, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_U_3), 3, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_U_4), 4, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_U_5), 5, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_U_6), 6, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_U_7), 7, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_U_8), 8, 39, 1, 2, 0, -1, 0.05f) \
        XAS(SID(PLAYER_ROLL_U_9), 9) \
        \
        XA(SID(PLAYER_ROLL_D_0), 9, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_D_1), 10, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_D_2), 11, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_D_3), 12, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_D_4), 13, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_D_5), 14, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_D_6), 15, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_D_7), 16, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_D_8), 17, 39, 1, 2, 0, -1, 0.05f) \
        XAS(SID(PLAYER_ROLL_D_9), 9) \
        \
        XA(SID(PLAYER_ROLL_R_0), 18, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_R_1), 19, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_R_2), 20, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_R_3), 21, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_R_4), 22, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_R_5), 21, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_R_6), 22, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_R_7), 23, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_R_8), 24, 39, 1, 2, 0, -1, 0.05f) \
        XAS(SID(PLAYER_ROLL_R_9), 9) \
        \
        XA(SID(PLAYER_ROLL_L_0), 25, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_L_1), 26, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_L_2), 27, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_L_3), 28, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_L_4), 29, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_L_5), 30, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_L_6), 31, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_L_7), 32, 39, 1, 2, 0, -1, 0.05f) \
        XA(SID(PLAYER_ROLL_L_8), 33, 39, 1, 2, 0, -1, 0.05f) \
        XAS(SID(PLAYER_ROLL_L_9), 9)

/* the spriteinfo_id_t corresponds to the index in the spriteinfo table */
typedef uint32_t spriteinfo_id_t;
typedef struct __attribute__((packed)) {
        uint16_t tx;
        uint16_t ty;
        uint16_t tw;
        uint16_t th;
} spriteinfo_t;

typedef vec2f32_t spriteinfo_bboff_t;

#define X(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff) id,
#define XA(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff, animlen) id,
#define XAS(id, animlen) id,
enum {
        X_SPRITEINFO_LIST
        SPRITEINFOS /* not beautiful, but little hack since __countof is unsupported in nvcc */
}; /* spriteinfo_id_t */
#undef X
#undef XA
#undef XAS

#define SPRITEINFO(tx, ty, comp_w, comp_h) {tx, ty, comp_w, comp_h}
#define SPRITEINFO_BBOFF(bb_xoff, bb_yoff) {bb_xoff, bb_yoff}

spriteinfo_t* spriteinfo_devtable_create();
float32_t* spriteinfo_animlen_devtable_create();
spriteinfo_bboff_t* spriteinfo_bboff_devtable_create();

void spriteinfo_devtable_destroy(spriteinfo_t* spriteinfo_devtable);
void spriteinfo_animlen_devtable_destroy(float32_t* spriteinfo_animlen_devtable);
void spriteinfo_bboff_devtable_destroy(spriteinfo_bboff_t* spriteinfo_bboff_devtable);

#ifdef __cplusplus
}
#endif

#endif