
#include "tileinfo.h"
#include "../def.h"
#include "util/cudamem.h"

static tileinfo_t* tileinfo_devtable;
static float32_t* tileinfo_animlen_devtable;

#define X_TILEINFO_TUPLE(id, tx, ty) TILEINFO(tx, ty),
#define X_TILEINFO_TUPLE_ANIM(id, tx, ty, animlen) TILEINFO(tx, ty), 
static const tileinfo_t tileinfo_hosttable[TILEINFOS] = {
        TILEINFO_LIST
};
#undef X_TILEINFO_TUPLE
#undef X_TILEINFO_TUPLE_ANIM

#define X_TILEINFO_TUPLE(id, tx, ty) [id] = 0.0f,
#define X_TILEINFO_TUPLE_ANIM(id, tx, ty, animlen) [id] = animlen,
static const float32_t tileinfo_animlen_hosttable[TILEINFOS] = {
        TILEINFO_LIST
};
#undef X_TILEINFO_TUPLE
#undef X_TILEINFO_TUPLE_ANIM

static inline void _devtable_alloc() {
        cudamem_alloc(&tileinfo_devtable, TILEINFOS * sizeof(tileinfo_t));
        if (sizeof(tileinfo_hosttable) != TILEINFOS * sizeof(tileinfo_t)) THROW("Texture devtable size mismatch");
        if (sizeof(tileinfo_hosttable) / sizeof(tileinfo_t) > TILEINFOS) THROW("Too many tileinfos for devtable");
        cudamem_copy(tileinfo_devtable, tileinfo_hosttable, TILEINFOS * sizeof(tileinfo_t), 1);
}

static inline void _animlen_devtable_alloc() {
        cudamem_alloc(&tileinfo_animlen_devtable, TILEINFOS * sizeof(float32_t));
        if (sizeof(tileinfo_animlen_hosttable) != TILEINFOS * sizeof(float32_t)) THROW("Animation length devtable size mismatch");
        if (sizeof(tileinfo_animlen_hosttable) / sizeof(float32_t) > TILEINFOS) THROW("Too many tileinfos for devtable");
        cudamem_copy(tileinfo_animlen_devtable, tileinfo_animlen_hosttable, TILEINFOS * sizeof(float32_t), 1);
}


void tileinfo_devtables_init() {
        _animlen_devtable_alloc();
        _devtable_alloc();

}

static inline void _animlen_devtable_free() {
        cudamem_free(tileinfo_animlen_devtable);
        tileinfo_animlen_devtable = 0;
}

static inline void _devtable_free() {
        cudamem_free(tileinfo_devtable);
        tileinfo_devtable = 0;
}

void tileinfo_devtables_cleanup() {
        _devtable_free();
        _animlen_devtable_free();
}

tileinfo_t* tileinfo_get_devtable() {
        return tileinfo_devtable;
}

float32_t* tileinfo_get_animlen_devtable() {
        return tileinfo_animlen_devtable;
}