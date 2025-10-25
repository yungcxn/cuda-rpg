#ifndef TILEINFO_H
#define TILEINFO_H

#include "../types/vec.h"

#define TILEINFO_ANIM_SENTINEL -1.0f

/* X_TILEINFO_TUPLE(tileinfo_id, tx, ty) */
/* X_TILEINFO_TUPLE_ANIM(tileinfo_id, tx, ty, framelen_in_seconds) */
/*   (ids of animation belonging together MUST be consecutive and end with TILEINFO_ANIM_SENTINEL) */
/* X_TILE_INFO_COMP(tileinfo_id, tx, ty, comp_w, comp_h) */
/* X_TILEINFO_TUPLE_COMP_ANIM(tileinfo_id, tx, ty, comp_w, comp_h, framelen_in_seconds) */

#define TILEINFO_LIST \
                  X_TILEINFO_TUPLE(TILEINFO_ID_GRASS1, 0, 0) \
                  X_TILEINFO_TUPLE(TILEINFO_ID_GRASS2, 1, 0) \
             X_TILEINFO_TUPLE_ANIM(TILEINFO_ID_WATER0, 10, 2, 1.0f) \
             X_TILEINFO_TUPLE_ANIM(TILEINFO_ID_WATER1, 10, 3, 1.0f) \
             X_TILEINFO_TUPLE_ANIM(TILEINFO_ID_WATER2, 10, 4, 1.0f) \
             X_TILEINFO_TUPLE_ANIM(TILEINFO_ID_WATER_END, 10, 5, TILEINFO_ANIM_SENTINEL)



/* the tileinfo_id_t corresponds to the index in the tileinfo table */
typedef uint16_t tileinfo_id_t;

typedef struct __attribute__((packed)) {
        uint16_t tx;
        uint16_t ty;
} tileinfo_t;

#define X_TILEINFO_TUPLE(id, tx, ty) id,
#define X_TILEINFO_TUPLE_ANIM(id, tx, ty, anim_speed) id,
#define X_TILEINFO_TUPLE_COMP(id, tx, ty, comp_w, comp_h) id,
#define X_TILEINFO_TUPLE_COMP_ANIM(id, tx, ty, comp_w, comp_h, anim_speed) id,
enum : tileinfo_id_t{
        TILEINFO_LIST
        TILEINFOS /* not beautiful, but little hack since __countof is unsupported in nvcc */
};
#undef X_TILEINFO_TUPLE
#undef X_TILEINFO_TUPLE_ANIM
#undef X_TILEINFO_TUPLE_COMP
#undef X_TILEINFO_TUPLE_COMP_ANIM

#define TILEINFO(tx, ty) {tx, ty}

#ifdef __cplusplus
extern "C" {
#endif

void   tileinfo_devtables_init();
void   tileinfo_devtables_cleanup();
tileinfo_t* tileinfo_get_devtable();
float32_t* tileinfo_get_animlen_devtable();

#ifdef __cplusplus
}
#endif

#endif