#ifndef DEF_H
#define DEF_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


#ifdef __CUDACC__ /* nvcc with c++?? */
typedef float  float32_t;
typedef double float64_t;
#else /* gcc with c23 */
typedef _Float32 float32_t;
typedef _Float64 float64_t;
#endif

#define BLOCK(X) do { \
        X \
} while(0)

#define THROW(fmt, ...) BLOCK (\
        fprintf(stderr, "Error at %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        exit(EXIT_FAILURE); \
)

#ifdef DEBUG
#define DEBUG_PRINT(fmt, ...) BLOCK (\
    fprintf(stdout, "Debug: " fmt "\n", ##__VA_ARGS__); \
)
#else
#define DEBUG_PRINT(fmt, ...) ((void)0)
#endif

#define NIBBLEPACK(hi, lo) ((((hi) & 0xF) << 4) | ((lo) & 0xF))

#define BIT8(n)  ((uint8_t)(1U << (n)))
#define BIT16(n) ((uint16_t)(1U << (n)))
#define BIT32(n) ((uint32_t)(1U << (n)))
#define BIT64(n) ((uint64_t)(1ULL << (n)))


#endif