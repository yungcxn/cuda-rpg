#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include <stdint.h>

#define WARP 32

#ifdef __cplusplus
extern "C" {
#endif

void kernel_draw(cudaSurfaceObject_t surface, uint32_t w, uint32_t h);

#ifdef __cplusplus
}
#endif

#endif
