
#include "kernel.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

__global__ static void _fill_chessboard_kernel(cudaSurfaceObject_t surface, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
                bool is_white = ((x + y) % 2 == 0);

                uchar4 color;
                if (is_white) {
                        color = make_uchar4(255, 255, 255, 255);
                } else {
                        color = make_uchar4(0, 0, 0, 255);
                }

                surf2Dwrite(color, surface, x * sizeof(uchar4), y);
        }
}

void kernel_draw(cudaSurfaceObject_t surf, uint32_t w, uint32_t h) {
        /* per-pixel kernel */
        dim3 block(16, 16);
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

        _fill_chessboard_kernel<<<grid, block>>>(surf, w, h);
}