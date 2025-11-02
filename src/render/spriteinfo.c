
#include "spriteinfo.h"
#include "../headeronly/def.h"
#include "util/ccuda.h"

#define X_SPRITEINFO_TUPLE(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff) \
        SPRITEINFO(tx, ty, comp_w, comp_h),
#define X_SPRITEINFO_TUPLE_ANIM(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff, animlen) \
        SPRITEINFO(tx, ty, comp_w, comp_h),
static const spriteinfo_t spriteinfo_hosttable[SPRITEINFOS] = {
        SPRITEINFO_LIST
};
#undef X_SPRITEINFO_TUPLE
#undef X_SPRITEINFO_TUPLE_ANIM

#define X_SPRITEINFO_TUPLE(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff) \
        [id] = SPRITEINFO_ANIM_NOANIMDURATION,
#define X_SPRITEINFO_TUPLE_ANIM(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff, animlen) \
        [id] = animlen,
static const float32_t spriteinfo_animlen_hosttable[SPRITEINFOS] = {
        SPRITEINFO_LIST
};
#undef X_SPRITEINFO_TUPLE
#undef X_SPRITEINFO_TUPLE_ANIM

#define X_SPRITEINFO_TUPLE(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff) \
        SPRITEINFO_BBOFF(bb_xoff, bb_yoff),
#define X_SPRITEINFO_TUPLE_ANIM(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff, animlen) \
        SPRITEINFO_BBOFF(bb_xoff, bb_yoff),
static const spriteinfo_bboff_t spriteinfo_bboff_hosttable[SPRITEINFOS] = {
        SPRITEINFO_LIST
};
#undef X_SPRITEINFO_TUPLE
#undef X_SPRITEINFO_TUPLE_ANIM

spriteinfo_t* spriteinfo_devtable_create() {
        spriteinfo_t* spriteinfo_devtable;
        ccuda_malloc((void**) &spriteinfo_devtable, SPRITEINFOS * sizeof(spriteinfo_t));
        if (sizeof(spriteinfo_hosttable) != SPRITEINFOS * sizeof(spriteinfo_t)) 
                THROW("Texture devtable size mismatch");
        if (sizeof(spriteinfo_hosttable) / sizeof(spriteinfo_t) > SPRITEINFOS) 
                THROW("Too many spriteinfos for devtable");
        ccuda_copy(
                spriteinfo_devtable, spriteinfo_hosttable,
                SPRITEINFOS * sizeof(spriteinfo_t), 1
        );
        return spriteinfo_devtable;
}

float32_t* spriteinfo_animlen_devtable_create() {
        float32_t* spriteinfo_animlen_devtable;
        ccuda_malloc((void**) &spriteinfo_animlen_devtable, SPRITEINFOS * sizeof(float32_t));
        if (sizeof(spriteinfo_animlen_hosttable) != SPRITEINFOS * sizeof(float32_t)) 
                THROW("Animation length devtable size mismatch");
        if (sizeof(spriteinfo_animlen_hosttable) / sizeof(float32_t) > SPRITEINFOS) 
                THROW("Too many spriteinfos for devtable");
        ccuda_copy(
                spriteinfo_animlen_devtable, spriteinfo_animlen_hosttable,
                SPRITEINFOS * sizeof(float32_t), 1
        );
        return spriteinfo_animlen_devtable;
}

spriteinfo_bboff_t* spriteinfo_bboff_devtable_create() {
        spriteinfo_bboff_t* spriteinfo_bboff_devtable;
        ccuda_malloc((void**) &spriteinfo_bboff_devtable, SPRITEINFOS * sizeof(spriteinfo_bboff_t));
        if (sizeof(spriteinfo_bboff_hosttable) != SPRITEINFOS * sizeof(spriteinfo_bboff_t)) 
                THROW("Bounding box offset devtable size mismatch");
        if (sizeof(spriteinfo_bboff_hosttable) / sizeof(spriteinfo_bboff_t) > SPRITEINFOS) 
                THROW("Too many spriteinfos for devtable");
        ccuda_copy(
                spriteinfo_bboff_devtable, spriteinfo_bboff_hosttable,
                SPRITEINFOS * sizeof(spriteinfo_bboff_t), 1
        );
        return spriteinfo_bboff_devtable;
}

void spriteinfo_devtable_free(spriteinfo_t* spriteinfo_devtable) {
        ccuda_free(spriteinfo_devtable);
        spriteinfo_devtable = 0;
}

void spriteinfo_animlen_devtable_free(float32_t* spriteinfo_animlen_devtable) {
        ccuda_free(spriteinfo_animlen_devtable);
        spriteinfo_animlen_devtable = 0;
}

void spriteinfo_bboff_devtable_free(spriteinfo_bboff_t* spriteinfo_bboff_devtable) {
        ccuda_free(spriteinfo_bboff_devtable);
        spriteinfo_bboff_devtable = 0;
}
