
#include "cudakernel.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include "../def.h"

__global__ static void _fill_test_kernel(cudaSurfaceObject_t surface, uint32_t width, uint32_t height) {
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x < width && y < height) {
                uchar4 color = make_uchar4(255, 0, 0, 255);
                surf2Dwrite<uchar4>(color, surface, x * sizeof(uchar4), y);
        }
}

__global__ static void _clear_overlay_kernel(cudaSurfaceObject_t surface, uint32_t width, uint32_t height) {
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
                uchar4 color = make_uchar4(0, 0, 0, 0);
                surf2Dwrite(color, surface, x * 4, y);
        }
}

void kernel_clear_overlay(cudaSurfaceObject_t surf, uint32_t w, uint32_t h) {
        dim3 block(16, 16);
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        _clear_overlay_kernel<<<grid, block>>>(surf, w, h);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) THROW("CUDA kernel error: %s\n", cudaGetErrorString(err));
}

void kernel_draw(cudaSurfaceObject_t surf, uint32_t w, uint32_t h) {
        dim3 block(16, 16);
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        _fill_test_kernel<<<grid, block>>>(surf, w, h);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) THROW("CUDA kernel error: %s\n", cudaGetErrorString(err));
}