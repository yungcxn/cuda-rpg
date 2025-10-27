#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WARP 32

void kernel_draw(cudaSurfaceObject_t surface, uint32_t w, uint32_t h);
void kernel_clear_overlay(cudaSurfaceObject_t surface, uint32_t w, uint32_t h);

#ifdef __cplusplus
}
#endif

#endif
