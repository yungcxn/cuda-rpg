#ifndef CUARRAY_CUH
#define CUARRAY_CUH

#include <cuda.h>

template <typename T>
struct cuarray_t {
    T* data;
    uint32_t length;
    uint32_t elemcount;
};

template <typename T>
__host__ cuarray_t<T>* cuarray_create(uint32_t length) {
        T* d_data;
        cudaMalloc((void**)&d_data, sizeof(T) * length);
        cuarray_t<T> h_cuarr = {
                .data = d_data,
                .length = length,
                .elemcount = 0
        };
        cuarray_t<T>* d_cuarr;
        cudaMalloc((void**)&d_cuarr, sizeof(cuarray_t<T>));
        cudaMemcpy(d_cuarr, &h_cuarr, sizeof(cuarray_t<T>), cudaMemcpyHostToDevice);
        return d_cuarr;
}

template <typename T>
__host__ __forceinline__ void cuarray_destroy(cuarray_t<T>* cuarr) {
        cudaFree((void*)cuarr->data);
        cudaFree((void*)cuarr);
}

template <typename T>
__host__ __forceinline__ void cuarray_hostclear(cuarray_t<T>* cuarr) {
        cudaMemset(&(cuarr->elemcount), 0, sizeof(uint32_t));
}

template <typename T>
__device__ __forceinline__ void cuarray_concadd(cuarray_t<T>* cuarr, const T val) {
        uint32_t idx = atomicAdd(&(cuarr->elemcount), 1);
        if (idx < cuarr->length) {
                cuarr->data[idx] = val;
        }
}

template <typename T>
__device__ __forceinline__ T* cuarray_getptr(cuarray_t<T>* cuarr, uint32_t idx) {
        if (idx >= cuarr->length) return 0;
        return &(cuarr->data[idx]);
}

template <typename T>
__device__ __forceinline__ T cuarray_get(cuarray_t<T>* cuarr, uint32_t idx) {
        return (idx < cuarr->elemcount) ? cuarr->data[idx] : T{};
}

template <typename T>
__device__ __forceinline__ uint32_t cuarray_length(cuarray_t<T>* cuarr) {
        return cuarr->elemcount;
}

template <typename T>
__host__ __forceinline__ uint32_t cuarray_hostlength(cuarray_t<T>* cuarr) {
        cuarray_t<T> h_cuarr;
        cudaMemcpy(&h_cuarr, cuarr, sizeof(cuarray_t<T>), cudaMemcpyDeviceToHost);
        return h_cuarr.elemcount;
}

#endif