
#include "tex.h"
#include "../res/res.h"
#include "util/ccuda.h"

tex_tileline_t* tex_devtilemap_create() {
        tex_tileline_t* devtilemap;
        const tex_tileline_t hosttilemap[] = { RES_MOTHERSHEET_DATA };
        ccuda_malloc(&devtilemap, sizeof(hosttilemap));
        ccuda_copy(devtilemap, hosttilemap, sizeof(hosttilemap), 1);
        return devtilemap;
}

tex_realrgba_t* tex_devpalette_create() {
        tex_realrgba_t* devpalette;
        const tex_realrgba_t hostpalette[] = { RES_PALETTE_DATA };
        ccuda_malloc(&devpalette, sizeof(hostpalette));
        ccuda_copy(devpalette, hostpalette, sizeof(hostpalette), 1);
}

void tex_devtilemap_destroy(tex_tileline_t* devtilemap) {
        ccuda_free(devtilemap);
        devtilemap = 0;
}

void tex_devpalette_destroy(tex_realrgba_t* devpalette) {
        ccuda_free(devpalette);
        devpalette = 0;
}
