#ifndef SPRITEINFO_H
#define SPRITEINFO_H

#include "../types/vec.h"

#define SPRITEINFO_ANIM_SENTINEL -1.0f

#define SPRITEINFO_LIST \
             X_SPRITEINFO_TUPLE(SPRITEINFO_ID_CHEST, 20, 0, 2, 1, 0, 0) \
        X_SPRITEINFO_TUPLE_ANIM(SPRITEINFO_ID_PLAYER0, 100, 0, 1, 2, 0, 0, 1.0f) \
        X_SPRITEINFO_TUPLE_ANIM(SPRITEINFO_ID_PLAYER1, 101, 0, 1, 2, 0, 0, 1.0f) \
        X_SPRITEINFO_TUPLE_ANIM(SPRITEINFO_ID_PLAYER_END, 102, 0, 1, 2, 0, 0, SPRITEINFO_ANIM_SENTINEL) 



/* the spriteinfo_id_t corresponds to the index in the spriteinfo table */
typedef uint16_t  spriteinfo_id_t;
typedef struct __attribute__((packed)) {
        uint16_t tx;
        uint16_t ty;
        uint16_t tw;
        uint16_t th;
} spriteinfo_t;

typedef vec2u16_t spriteinfo_bboff_t;

#define X_SPRITEINFO_TUPLE(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff) id,
#define X_SPRITEINFO_TUPLE_ANIM(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff, anim_speed) id,
enum : spriteinfo_id_t{
        SPRITEINFO_LIST
        SPRITEINFOS /* not beautiful, but little hack since __countof is unsupported in nvcc */
};
#undef X_SPRITEINFO_TUPLE
#undef X_SPRITEINFO_TUPLE_ANIM

#define SPRITEINFO(tx, ty, comp_w, comp_h) {tx, ty, comp_w, comp_h}
#define SPRITEINFO_BBOFF(bb_xoff, bb_yoff) {bb_xoff, bb_yoff}

#ifdef __cplusplus
extern "C" {
#endif

void  spriteinfo_devtables_init();
void  spriteinfo_devtables_cleanup();
spriteinfo_t*       spriteinfo_get_devtable();
float32_t*          spriteinfo_get_animlen_devtable();
spriteinfo_bboff_t* spriteinfo_get_bboff_devtable();

#ifdef __cplusplus
}
#endif

#endif