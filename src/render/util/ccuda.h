
#ifndef CCUDA_H

#define CCUDA_H

#include <stdint.h>
#include <stddef.h>

typedef unsigned long long ccuda_surfaceobj_t;

/* types may be shared, so used by .cuh, but functions should be natively used */
#ifndef __CUDACC__
/* Allocate device memory */
uint32_t ccuda_malloc(void** ptr, size_t bytes);
uint32_t ccuda_mallochost(void** ptr, size_t bytes);

/* Free device memory */
uint32_t ccuda_free(void* ptr);
uint32_t ccuda_freehost(void* ptr);

/* Copy: 0=H2H, 1=H2D, 2=D2H, 3=D2D */
uint32_t ccuda_copy(void* dst, const void* src, size_t bytes, uint32_t direction);

/* Move: alloc new, copy, free old */
uint32_t ccuda_move(void** dst, const void* src, size_t bytes, uint32_t direction);

uint32_t ccuda_memset(void* dst, uint32_t value, size_t bytes);

#endif

#endif
