
#include <cuda.h>
#include <cuda_runtime.h>
#include "tileinfo.h"
#include "../def.h"

static tileinfo_t* tileinfo_devtable;
static float32_t* tileinfo_animlen_devtable;

static void _devtable_alloc() {
        cudaMalloc(&tileinfo_devtable, TILEINFOS * sizeof(tileinfo_t));

        #define X_TILEINFO_TUPLE(id, tx, ty) TILEINFO(tx, ty),
        #define X_TILEINFO_TUPLE_ANIM(id, tx, ty, animlen) TILEINFO(tx, ty),
        tileinfo_t tocopy[TILEINFOS] = {
                TILEINFO_LIST
        };
        #undef X_TILEINFO_TUPLE
        #undef X_TILEINFO_TUPLE_ANIM

        if (sizeof(tocopy) != TILEINFOS * sizeof(tileinfo_t)) THROW("Texture devtable size mismatch");
        if (sizeof(tocopy) / sizeof(tileinfo_t) > TILEINFOS) THROW("Too many tileinfos for devtable");

        cudaMemcpy(tileinfo_devtable, tocopy, TILEINFOS * sizeof(tileinfo_t), cudaMemcpyHostToDevice);
}

static void _animlen_devtable_alloc() {
        cudaMalloc(&tileinfo_animlen_devtable, TILEINFOS * sizeof(float32_t));
        #define X_TILEINFO_TUPLE(id, tx, ty) [id] = 1.0f,
        #define X_TILEINFO_TUPLE_ANIM(id, tx, ty, animlen) [id] = animlen,
        float32_t tocopy[TILEINFOS] = {
                TILEINFO_LIST
        };
        #undef X_TILEINFO_TUPLE
        #undef X_TILEINFO_TUPLE_ANIM

        if (sizeof(tocopy) != TILEINFOS * sizeof(float32_t)) THROW("Animation length devtable size mismatch");
        if (sizeof(tocopy) / sizeof(float32_t) > TILEINFOS) THROW("Too many tileinfos for devtable");

        cudaMemcpy(tileinfo_animlen_devtable, tocopy, TILEINFOS * sizeof(float32_t), cudaMemcpyHostToDevice);
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