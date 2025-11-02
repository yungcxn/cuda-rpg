#include "ccuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

extern "C" {

uint32_t ccuda_malloc(void** ptr, size_t bytes) {
        cudaError_t err = cudaMalloc(ptr, bytes);
        if (err != cudaSuccess) {
                std::fprintf(stderr, "ccuda_nalloc: %s\n", cudaGetErrorString(err));
                *ptr = nullptr;
                return -1;
        }
        return 0;
}

uint32_t ccuda_mallochost(void** ptr, size_t bytes) {
        cudaError_t err = cudaMallocHost(ptr, bytes);
        if (err != cudaSuccess) {
                std::fprintf(stderr, "ccuda_mallochost: %s\n", cudaGetErrorString(err));
                *ptr = nullptr;
                return -1;
        }
        return 0;
}

uint32_t ccuda_free(void* ptr) {
        if (!ptr) return 0;
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
                std::fprintf(stderr, "ccuda_free: %s\n", cudaGetErrorString(err));
                return -1;
        }
        return 0;
}

uint32_t ccuda_freehost(void* ptr) {
        if (!ptr) return 0;
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess) {
                std::fprintf(stderr, "ccuda_freehost: %s\n", cudaGetErrorString(err));
                return -1;
        }
        return 0;
}

uint32_t ccuda_copy(void* dst, const void* src, size_t bytes, uint32_t direction) {
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
                std::fprintf(stderr, "ccuda_copy: %s\n", cudaGetErrorString(err));
                return -1;
        }
        return 0;
}

uint32_t ccuda_move(void** dst, const void* src, size_t bytes, uint32_t direction) {
        void* new_ptr = nullptr;
        if (ccuda_malloc(&new_ptr, bytes) != 0)
                return -1;
        if (ccuda_copy(new_ptr, src, bytes, direction) != 0) {
                ccuda_free(new_ptr);
                return -1;
        }
        if (*dst)
                ccuda_free(*dst);
        *dst = new_ptr;
        return 0;
}

uint32_t ccuda_memset(void* dst, uint32_t value, size_t bytes) {
        cudaError_t err = cudaMemset(dst, value, bytes);
        if (err != cudaSuccess) {
                std::fprintf(stderr, "ccuda_set: %s\n", cudaGetErrorString(err));
                return -1;
        }
        return 0;
}

}
