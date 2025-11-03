#include "ccuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#include "../../headeronly/def.h"

extern "C" {

void ccuda_malloc(void** ptr, size_t bytes) {
        cudaError_t err = cudaMalloc(ptr, bytes);
        if (err != cudaSuccess) THROW("ccuda_malloc: %s\n", cudaGetErrorString(err));
}

void ccuda_mallochost(void** ptr, size_t bytes) {
        cudaError_t err = cudaMallocHost(ptr, bytes);
        if (err != cudaSuccess) THROW("ccuda_mallochost: %s\n", cudaGetErrorString(err));
}

void ccuda_free(void* ptr) {
        if (!ptr) THROW("ccuda_free: null pointer\n");
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) THROW("ccuda_free: %s\n", cudaGetErrorString(err));
}

void ccuda_freehost(void* ptr) {
        if (!ptr) THROW("ccuda_freehost: null pointer\n");
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess) THROW("ccuda_freehost: %s\n", cudaGetErrorString(err));
}

void ccuda_copy(void* dst, const void* src, size_t bytes, uint32_t direction) {
        cudaMemcpyKind kind;
        switch (direction) {
                case 0: kind = cudaMemcpyHostToHost; break;
                case 1: kind = cudaMemcpyHostToDevice; break;
                case 2: kind = cudaMemcpyDeviceToHost; break;
                case 3: kind = cudaMemcpyDeviceToDevice; break;
                default: THROW("ccuda_copy: invalid direction %u\n", direction);
        }
        cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
        if (err != cudaSuccess) THROW("ccuda_copy: %s\n", cudaGetErrorString(err));
}

void ccuda_move(void** dst, const void* src, size_t bytes, uint32_t direction) {
        void* new_ptr = nullptr;
        ccuda_malloc(&new_ptr, bytes);
        ccuda_copy(new_ptr, src, bytes, direction);
        if (*dst)
                ccuda_free(*dst);
        *dst = new_ptr;
}

void ccuda_memset(void* dst, uint32_t value, size_t bytes) {
        cudaError_t err = cudaMemset(dst, value, bytes);
        if (err != cudaSuccess) THROW("ccuda_memset: %s\n", cudaGetErrorString(err));
}

}
