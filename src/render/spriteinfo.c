
#include "spriteinfo.h"
#include "../def.h"
#include "util/cudamem.h"

static spriteinfo_t* spriteinfo_devtable;
static float32_t* spriteinfo_animlen_devtable;
static spriteinfo_bboff_t* spriteinfo_bboff_devtable;

#define X_SPRITEINFO_TUPLE(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff) SPRITEINFO(tx, ty, comp_w, comp_h),
#define X_SPRITEINFO_TUPLE_ANIM(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff, animlen) SPRITEINFO(tx, ty, comp_w, comp_h),
static const spriteinfo_t spriteinfo_hosttable[SPRITEINFOS] = {
        SPRITEINFO_LIST
};
#undef X_SPRITEINFO_TUPLE
#undef X_SPRITEINFO_TUPLE_ANIM

#define X_SPRITEINFO_TUPLE(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff) [id] = 0.0f,
#define X_SPRITEINFO_TUPLE_ANIM(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff, animlen) [id] = animlen,
static const float32_t spriteinfo_animlen_hosttable[SPRITEINFOS] = {
        SPRITEINFO_LIST
};
#undef X_SPRITEINFO_TUPLE
#undef X_SPRITEINFO_TUPLE_ANIM

#define X_SPRITEINFO_TUPLE(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff) SPRITEINFO_BBOFF(bb_xoff, bb_yoff),
#define X_SPRITEINFO_TUPLE_ANIM(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff, animlen) SPRITEINFO_BBOFF(bb_xoff, bb_yoff),
static const spriteinfo_bboff_t spriteinfo_bboff_hosttable[SPRITEINFOS] = {
        SPRITEINFO_LIST
};
#undef X_SPRITEINFO_TUPLE
#undef X_SPRITEINFO_TUPLE_ANIM

static inline void _devtable_alloc() {
        cudamem_alloc(&spriteinfo_devtable, SPRITEINFOS * sizeof(spriteinfo_t));
        if (sizeof(spriteinfo_hosttable) != SPRITEINFOS * sizeof(spriteinfo_t)) THROW("Texture devtable size mismatch");
        if (sizeof(spriteinfo_hosttable) / sizeof(spriteinfo_t) > SPRITEINFOS) THROW("Too many spriteinfos for devtable");
        cudamem_alloc(spriteinfo_devtable, spriteinfo_hosttable, SPRITEINFOS * sizeof(spriteinfo_t), 1);
}

static inline void _animlen_devtable_alloc() {
        cudamem_alloc(&spriteinfo_animlen_devtable, SPRITEINFOS * sizeof(float32_t));
        if (sizeof(spriteinfo_animlen_hosttable) != SPRITEINFOS * sizeof(float32_t)) THROW("Animation length devtable size mismatch");
        if (sizeof(spriteinfo_animlen_hosttable) / sizeof(float32_t) > SPRITEINFOS) THROW("Too many spriteinfos for devtable");
        cudamem_copy(spriteinfo_animlen_devtable, spriteinfo_animlen_hosttable, SPRITEINFOS * sizeof(float32_t), 1);
}

static inline void _bboff_devtable_alloc() {
        cudamem_alloc(&spriteinfo_bboff_devtable, SPRITEINFOS * sizeof(spriteinfo_bboff_t));
        if (sizeof(spriteinfo_bboff_hosttable) != SPRITEINFOS * sizeof(spriteinfo_bboff_t)) THROW("Bounding box offset devtable size mismatch");
        if (sizeof(spriteinfo_bboff_hosttable) / sizeof(spriteinfo_bboff_t) > SPRITEINFOS) THROW("Too many spriteinfos for devtable");
        cudamem_copy(spriteinfo_bboff_devtable, spriteinfo_bboff_hosttable, SPRITEINFOS * sizeof(spriteinfo_bboff_t), 1);
}

void spriteinfo_devtables_init() {
        _animlen_devtable_alloc();
        _devtable_alloc();
        _bboff_devtable_alloc();
}

static inline void _animlen_devtable_free() {
        cudamem_free(spriteinfo_animlen_devtable);
        spriteinfo_animlen_devtable = 0;
}

static inline void _devtable_free() {
        cudamem_free(spriteinfo_devtable);
        spriteinfo_devtable = 0;
}

static inline void _bboff_devtable_free() {
        cudamem_free(spriteinfo_bboff_devtable);
        spriteinfo_bboff_devtable = 0;
}

void spriteinfo_devtables_cleanup() {
        _devtable_free();
        _animlen_devtable_free();
        _bboff_devtable_free();
}

spriteinfo_t* spriteinfo_get_devtable() {
        return spriteinfo_devtable;
}

float32_t* spriteinfo_get_animlen_devtable() {
        return spriteinfo_animlen_devtable;
}

spriteinfo_bboff_t* spriteinfo_get_bboff_devtable() {
        return spriteinfo_bboff_devtable;
}