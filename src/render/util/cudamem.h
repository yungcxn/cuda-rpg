#ifndef CUDAMEM_H
#define CUDAMEM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate device memory */
uint32_t   cudamem_alloc(void** ptr, size_t bytes);

/* Free device memory */
uint32_t   cudamem_free(void* ptr);

/* Copy: 0=H2H, 1=H2D, 2=D2H, 3=D2D */
uint32_t   cudamem_copy(void* dst, const void* src, size_t bytes, uint32_t direction);

/* Move: alloc new, copy, free old */
uint32_t   cudamem_move(void** dst, const void* src, size_t bytes, uint32_t direction);

#ifdef __cplusplus
}
#endif

#endif
