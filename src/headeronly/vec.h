#ifndef VEC_H
#define VEC_H

#include "../headeronly/def.h"
#include <stdint.h>

typedef struct __attribute__((packed)) {
        float32_t x;
        float32_t y;
} vec2f32_t;

typedef struct __attribute__((packed)) {
        uint32_t x;
        uint32_t y;
} vec2u32_t;

typedef struct __attribute__((packed)) {
        int32_t x;
        int32_t y;
} vec2i32_t;

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

static inline float32_t fast_rsqrt(float32_t number) {
    union { float32_t f; uint32_t i; } conv = { number };
    float32_t x2 = number * 0.5f;
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f = conv.f * (1.5f - (x2 * conv.f * conv.f));
    return conv.f;
}

static inline void vecf32_normalize(vec2f32_t* v) {
        float32_t len_sq = v->x * v->x + v->y * v->y;
        float32_t inv_len = fast_rsqrt(len_sq);
        v->x *= inv_len;
        v->y *= inv_len;
}


#endif