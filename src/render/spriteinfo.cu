
#include <cuda.h>
#include <cuda_runtime.h>
#include "spriteinfo.cuh"
#include "../def.h"

static spriteinfo_t* spriteinfo_devtable;
static float32_t* spriteinfo_animlen_devtable;
static spriteinfo_bboff_t* spriteinfo_bboff_devtable;

static void _devtable_alloc() {
        cudaMalloc(&spriteinfo_devtable, SPRITEINFOS * sizeof(spriteinfo_t));

        #define X_SPRITEINFO_TUPLE(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff) SPRITEINFO(tx, ty, comp_w, comp_h),
        #define X_SPRITEINFO_TUPLE_ANIM(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff, animlen) SPRITEINFO(tx, ty, comp_w, comp_h),
        spriteinfo_t tocopy[SPRITEINFOS] = {
                SPRITEINFO_LIST
        };
        #undef X_SPRITEINFO_TUPLE
        #undef X_SPRITEINFO_TUPLE_ANIM

        if (sizeof(tocopy) != SPRITEINFOS * sizeof(spriteinfo_t)) THROW("Texture devtable size mismatch");
        if (sizeof(tocopy) / sizeof(spriteinfo_t) > SPRITEINFOS) THROW("Too many spriteinfos for devtable");

        cudaMemcpy(spriteinfo_devtable, tocopy, SPRITEINFOS * sizeof(spriteinfo_t), cudaMemcpyHostToDevice);
}

static void _animlen_devtable_alloc() {
        cudaMalloc(&spriteinfo_animlen_devtable, SPRITEINFOS * sizeof(float32_t));
        #define X_SPRITEINFO_TUPLE(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff) [id] = 0.0f,
        #define X_SPRITEINFO_TUPLE_ANIM(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff, animlen) [id] = animlen,
        float32_t tocopy[SPRITEINFOS] = {
                SPRITEINFO_LIST
        };
        #undef X_SPRITEINFO_TUPLE
        #undef X_SPRITEINFO_TUPLE_ANIM

        if (sizeof(tocopy) != SPRITEINFOS * sizeof(float32_t)) THROW("Animation length devtable size mismatch");
        if (sizeof(tocopy) / sizeof(float32_t) > SPRITEINFOS) THROW("Too many spriteinfos for devtable");

        cudaMemcpy(spriteinfo_animlen_devtable, tocopy, SPRITEINFOS * sizeof(float32_t), cudaMemcpyHostToDevice);
}

static void _bboff_devtable_alloc() {
        cudaMalloc(&spriteinfo_bboff_devtable, SPRITEINFOS * sizeof(spriteinfo_bboff_t));
        #define X_SPRITEINFO_TUPLE(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff) SPRITEINFO_BBOFF(bb_xoff, bb_yoff),
        #define X_SPRITEINFO_TUPLE_ANIM(id, tx, ty, comp_w, comp_h, bb_xoff, bb_yoff, animlen) SPRITEINFO_BBOFF(bb_xoff, bb_yoff),
        spriteinfo_bboff_t tocopy[SPRITEINFOS] = {
                SPRITEINFO_LIST
        };
        #undef X_SPRITEINFO_TUPLE
        #undef X_SPRITEINFO_TUPLE_ANIM

        if (sizeof(tocopy) != SPRITEINFOS * sizeof(spriteinfo_bboff_t)) THROW("Bounding box offset devtable size mismatch");
        if (sizeof(tocopy) / sizeof(spriteinfo_bboff_t) > SPRITEINFOS) THROW("Too many spriteinfos for devtable");

        cudaMemcpy(spriteinfo_bboff_devtable, tocopy, SPRITEINFOS * sizeof(spriteinfo_bboff_t), cudaMemcpyHostToDevice);
}

void spriteinfo_devtables_init() {
        _animlen_devtable_alloc();
        _devtable_alloc();
        _bboff_devtable_alloc();

}

static void _animlen_devtable_free() {
        cudaFree(spriteinfo_animlen_devtable);
        spriteinfo_animlen_devtable = 0;
}

static void _devtable_free() {
        cudaFree(spriteinfo_devtable);
        spriteinfo_devtable = 0;
}

static void _bboff_devtable_free() {
        cudaFree(spriteinfo_bboff_devtable);
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