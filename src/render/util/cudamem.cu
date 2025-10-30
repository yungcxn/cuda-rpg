#include "cudamem.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

extern "C" {

uint32_t cudamem_alloc(void** ptr, size_t bytes) {
        cudaError_t err = cudaMalloc(ptr, bytes);
        if (err != cudaSuccess) {
                std::fprintf(stderr, "cudamem_alloc: %s\n", cudaGetErrorString(err));
                *ptr = nullptr;
                return -1;
        }
        return 0;
}

uint32_t cudamem_free(void* ptr) {
        if (!ptr) return 0;
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
                std::fprintf(stderr, "cudamem_free: %s\n", cudaGetErrorString(err));
                return -1;
        }
        return 0;
}

uint32_t cudamem_copy(void* dst, const void* src, size_t bytes, uint32_t direction) {
        cudaMemcpyKind kind;
        switch (direction) {
                case 0: kind = cudaMemcpyHostToHost; break;
                case 1: kind = cudaMemcpyHostToDevice; break;
                case 2: kind = cudaMemcpyDeviceToHost; break;
                case 3: kind = cudaMemcpyDeviceToDevice; break;
                default: return -1;
        }
        cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
        if (err != cudaSuccess) {
                std::fprintf(stderr, "cudamem_copy: %s\n", cudaGetErrorString(err));
                return -1;
        }
        return 0;
}

uint32_t cudamem_move(void** dst, const void* src, size_t bytes, uint32_t direction) {
        void* new_ptr = nullptr;
        if (cudamem_alloc(&new_ptr, bytes) != 0)
                return -1;
        if (cudamem_copy(new_ptr, src, bytes, direction) != 0) {
                cudamem_free(new_ptr);
                return -1;
        }
        if (*dst)
                cudamem_free(*dst);
        *dst = new_ptr;
        return 0;
}

}
