
#include <cuda.h>
#include <cuda_runtime.h>
#include "tileinfo.h"
#include "../def.h"

static tileinfo_t* tileinfo_devtable;
static float32_t* tileinfo_animlen_devtable;

#define X_TILEINFO_TUPLE(id, tx, ty) TILEINFO(tx, ty),
#define X_TILEINFO_TUPLE_ANIM(id, tx, ty, animlen) TILEINFO(tx, ty), 
constexpr static tileinfo_t tileinfo_hosttable[TILEINFOS] = {
        TILEINFO_LIST
};
#undef X_TILEINFO_TUPLE
#undef X_TILEINFO_TUPLE_ANIM

#define X_TILEINFO_TUPLE(id, tx, ty) [id] = 0.0f,
#define X_TILEINFO_TUPLE_ANIM(id, tx, ty, animlen) [id] = animlen,
constexpr static float32_t tileinfo_animlen_hosttable[TILEINFOS] = {
        TILEINFO_LIST
};
#undef X_TILEINFO_TUPLE
#undef X_TILEINFO_TUPLE_ANIM

static void _devtable_alloc() {
        cudaMalloc(&tileinfo_devtable, TILEINFOS * sizeof(tileinfo_t));
        if (sizeof(tileinfo_hosttable) != TILEINFOS * sizeof(tileinfo_t)) THROW("Texture devtable size mismatch");
        if (sizeof(tileinfo_hosttable) / sizeof(tileinfo_t) > TILEINFOS) THROW("Too many tileinfos for devtable");
        cudaMemcpy(tileinfo_devtable, tileinfo_hosttable, TILEINFOS * sizeof(tileinfo_t), cudaMemcpyHostToDevice);
}

static void _animlen_devtable_alloc() {
        cudaMalloc(&tileinfo_animlen_devtable, TILEINFOS * sizeof(float32_t));
        if (sizeof(tileinfo_animlen_hosttable) != TILEINFOS * sizeof(float32_t)) THROW("Animation length devtable size mismatch");
        if (sizeof(tileinfo_animlen_hosttable) / sizeof(float32_t) > TILEINFOS) THROW("Too many tileinfos for devtable");
        cudaMemcpy(tileinfo_animlen_devtable, tileinfo_animlen_hosttable, TILEINFOS * sizeof(float32_t), cudaMemcpyHostToDevice);
}


void tileinfo_devtables_init() {
        _animlen_devtable_alloc();
        _devtable_alloc();

}

static void _animlen_devtable_free() {
        cudaFree(tileinfo_animlen_devtable);
        tileinfo_animlen_devtable = 0;
}

static void _devtable_free() {
        cudaFree(tileinfo_devtable);
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