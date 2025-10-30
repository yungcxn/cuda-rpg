#ifndef VEC_H
#define VEC_H

#include <stdio.h>
#include <stdlib.h>
#include "../def.h"

typedef struct __attribute__((packed)) {
        float32_t x;
        float32_t y;
} vec2f32_t;

typedef struct __attribute__((packed)) {
        uint32_t x;
        uint32_t y;
} vec2u32_t;

typedef struct __attribute__((packed)) {
        uint16_t x;
        uint16_t y;
} vec2u16_t;

typedef struct __attribute__((packed)) {
        uint16_t a;
        uint16_t b;
        uint16_t c;
        uint16_t d;
} vec4u16_t;

#define VEC2(x, y) { x, y }
#define VEC4(a, b, c, d) { a, b, c, d }

static inline void vec2f32_add(vec2f32_t* a, vec2f32_t* b) {
        a->x += b->x;
        a->y += b->y;
}

static inline void vecf32_normalize(vec2f32_t* v) {
        float32_t len_sq = v->x * v->x + v->y * v->y;
        float32_t inv_len = 1.0f / sqrtf(len_sq + (len_sq == 0.0f));
        v->x *= inv_len;
        v->y *= inv_len;
}


#endif