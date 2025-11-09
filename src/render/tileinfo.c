
#include "tileinfo.h"
#include "../headeronly/def.h"
#include "util/ccuda.h"

#define X(id, tx, ty) TILEINFO(tx, ty),
#define XA(id, tx, ty, animlen) TILEINFO(tx, ty), 
#define XAS(id, animlen) TILEINFO(0, 0),
static const tileinfo_t tileinfo_hosttable[TILEINFOS] = {
        X_TILEINFO_LIST
};
#undef X
#undef XA
#undef XAS

#define X(id, tx, ty) [id] = 0.0f,
#define XA(id, tx, ty, animlen) [id] = animlen,
#define XAS(id, animlen) [id] = -animlen,
static const float32_t tileinfo_animlen_hosttable[TILEINFOS] = {
        X_TILEINFO_LIST
};
#undef X
#undef XA
#undef XAS

tileinfo_t* tileinfo_devtable_create(void) {
        tileinfo_t* tileinfo_devtable;
        ccuda_malloc((void**) &tileinfo_devtable, TILEINFOS * sizeof(tileinfo_t));
        if (sizeof(tileinfo_hosttable) != TILEINFOS * sizeof(tileinfo_t)) 
                THROW("Texture devtable size mismatch");
        if (sizeof(tileinfo_hosttable) / sizeof(tileinfo_t) > TILEINFOS) 
                THROW("Too many tileinfos for devtable");
        ccuda_copy(tileinfo_devtable, tileinfo_hosttable, TILEINFOS * sizeof(tileinfo_t), 1);
        return tileinfo_devtable;
}

void tileinfo_animlen_devtable_destroy(float32_t* tileinfo_animlen_devtable) {
        ccuda_free(tileinfo_animlen_devtable);
        tileinfo_animlen_devtable = 0;
}

float32_t* tileinfo_animlen_devtable_create(void) {
        float32_t* tileinfo_animlen_devtable;
        ccuda_malloc((void**) &tileinfo_animlen_devtable, TILEINFOS * sizeof(float32_t));
        if (sizeof(tileinfo_animlen_hosttable) != TILEINFOS * sizeof(float32_t)) 
                THROW("Animation length devtable size mismatch");
        if (sizeof(tileinfo_animlen_hosttable) / sizeof(float32_t) > TILEINFOS) 
                THROW("Too many tileinfos for devtable");
        ccuda_copy(
                tileinfo_animlen_devtable, tileinfo_animlen_hosttable,
                TILEINFOS * sizeof(float32_t), 1
        );
        return tileinfo_animlen_devtable;
}

void tileinfo_devtable_destroy(tileinfo_t* tileinfo_devtable) {
        ccuda_free(tileinfo_devtable);
        tileinfo_devtable = 0;
}

float32_t* tileinfo_animtimer_devcache_create(void) {
        float32_t* tileinfo_animtimer_devcache;
        ccuda_malloc((void**) &tileinfo_animtimer_devcache, TILEINFOS * sizeof(float32_t));
        ccuda_memset(tileinfo_animtimer_devcache, 0, TILEINFOS * sizeof(float32_t));
        return tileinfo_animtimer_devcache;
}

void tileinfo_animtimer_devcache_destroy(float32_t* tileinfo_animtimer_devcache) {
        ccuda_free(tileinfo_animtimer_devcache);
        tileinfo_animtimer_devcache = 0;
}
