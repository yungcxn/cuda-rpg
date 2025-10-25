#include <cuda.h>
#include <cuda_runtime.h>
#include "tex.h"
#include "../res/res.h"

static tex_tileline_t* devtilemap;
static tex_realrgba_t* devpalette;

static void _devtilemap_alloc() {
        const tex_tileline_t hosttilemap[] = { RES_MOTHERSHEET_DATA };
        cudaMalloc(&devtilemap, sizeof(hosttilemap));
        cudaMemcpy(devtilemap, hosttilemap, sizeof(hosttilemap), cudaMemcpyHostToDevice);
}

static void _devpalette_alloc() {
        const tex_realrgba_t hostpalette[] = { RES_PALETTE_DATA };
        cudaMalloc(&devpalette, sizeof(hostpalette));
        cudaMemcpy(devpalette, hostpalette, sizeof(hostpalette), cudaMemcpyHostToDevice);
}

void tex_devdata_init() {
        _devtilemap_alloc();
        _devpalette_alloc();
}

static void _devtilemap_free() {
        cudaFree(devtilemap);
        devtilemap = 0;
}

static void _devpalette_free() {
        cudaFree(devpalette);
        devpalette = 0;
}

void tex_devdata_cleanup() {
        _devtilemap_free();
        _devpalette_free();
}