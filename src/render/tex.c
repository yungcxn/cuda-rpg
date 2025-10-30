
#include "tex.h"
#include "../res/res.h"
#include "util/cudamem.h"

static tex_tileline_t* devtilemap;
static tex_realrgba_t* devpalette;

static inline void _devtilemap_alloc() {
        const tex_tileline_t hosttilemap[] = { RES_MOTHERSHEET_DATA };
        cudamem_alloc(&devtilemap, sizeof(hosttilemap));
        cudamem_copy(devtilemap, hosttilemap, sizeof(hosttilemap), 1);
}

static inline void _devpalette_alloc() {
        const tex_realrgba_t hostpalette[] = { RES_PALETTE_DATA };
        cudamem_alloc(&devpalette, sizeof(hostpalette));
        cudamem_copy(devpalette, hostpalette, sizeof(hostpalette), 1);
}

void tex_devdata_init() {
        _devtilemap_alloc();
        _devpalette_alloc();
}

static inline void _devtilemap_free() {
        cudamem_free(devtilemap);
        devtilemap = 0;
}

static inline void _devpalette_free() {
        cudamem_free(devpalette);
        devpalette = 0;
}

void tex_devdata_cleanup() {
        _devtilemap_free();
        _devpalette_free();
}