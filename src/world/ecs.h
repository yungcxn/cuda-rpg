#ifndef ECS_H
#define ECS_H

#include <stdint.h>
#include "../headeronly/vec.h"
#include "../render/spriteinfo.h"

#define ECS_MAX_ENTITIES 1024
#define ECS_ENTITY_DEADMASK BIT64(63)

#define ECS_ISDEAD(pos1) ((pos1).x == -1.0f && (pos1).y == -1.0f)
#define ECS_GARBAGE_REMOVAL_THRESHOLD 100  /* number of killed entities to trigger garbage removal */
#define _UINT32_IN_AVX2REG 8

#define ECS_ENTITYTYPE_LIST \
        X_ENTITYTYPE_TUPLE(ECS_ENTITY_ZOMBIE, entityzombie_update)

#define X_ENTITYTYPE_TUPLE(name_id, updatefunc) name_id,
typedef enum {
        ECS_ENTITYTYPE_LIST
        ECS_ENTITYTYPE_COUNT
} ecs_entitytype_t;
#undef X_ENTITYTYPE_TUPLE

typedef void (*ecs_entity_updatefunc_t)(uint32_t entity_id, float32_t dt);

typedef struct __attribute__((aligned(32))) {
        spriteinfo_id_t spriteinfo[ECS_MAX_ENTITIES]; /* + 32 => 400 */
        float32_t spritetimers[ECS_MAX_ENTITIES]; /* + 32 => 432 */
} ecs_shared_t; /* this is shared between dev and host */

typedef struct __attribute__((aligned(32))) {
        uint64_t state_hi[ECS_MAX_ENTITIES]; /* 64 */
        uint64_t state_lo[ECS_MAX_ENTITIES]; /* + 64 => 128 */
        vec2f32_t pos1[ECS_MAX_ENTITIES]; /* + 64 => 192 */
        vec2f32_t pos2[ECS_MAX_ENTITIES]; /* + 64 => 256 */
        vec2f32_t velocity[ECS_MAX_ENTITIES]; /* + 64 => 320 */
        /* we handle sprites on GPU */
        ecs_entitytype_t entitytype[ECS_MAX_ENTITIES]; /* + 32 => 352 */
        uint32_t root_entity_id[ECS_MAX_ENTITIES]; /* + 32 => 384 */
        ecs_shared_t* shared; /* + 64 => 432 */
} ecs_t;

typedef struct __attribute__((aligned(32))) {
        ecs_t* instance;
        uint16_t living_count;
        uint16_t dead_count;
} ecs_handle_t;

typedef struct { /* unpacked */
        vec2f32_t pos1;
        vec2f32_t pos2;
        ecs_entitytype_t entitytype;
        uint32_t root_entity_id;
        spriteinfo_id_t spriteinfo;
} ecs_exec_spawndata_t;

typedef struct { /* unpacked */
        vec2f32_t pos1[_UINT32_IN_AVX2REG]; /* 2 simd regs */
        vec2f32_t pos2[_UINT32_IN_AVX2REG]; /* 2 simd regs */
        ecs_entitytype_t entitytype[_UINT32_IN_AVX2REG]; /* 1 simd reg */
        uint32_t root_entity_id[_UINT32_IN_AVX2REG]; /* 1 simd reg */
        spriteinfo_id_t spriteinfo[_UINT32_IN_AVX2REG]; /* 1 simd reg */
} ecs_exec_spawndata_avx2_t;

typedef struct { /* unpacked */
        uint32_t entity_id;
} ecs_exec_killdata_t;

typedef struct { /* unpacked */
        uint32_t entity_startid8x;
} ecs_exec_killdata_avx2_t;

typedef struct { /* unpacked */
        uint32_t entity_id;
        vec2f32_t velocity_diff;
} ecs_exec_velocity_acceldata_t;

typedef struct { /* unpacked */
        uint32_t entity_startid8x;
        vec2f32_t velocity_diff[_UINT32_IN_AVX2REG]; /* 2 simd regs */
} ecs_exec_velocity_acceldata_avx2_t;

typedef struct { /* unpacked */
        uint32_t entity_id;
        vec2f32_t velocity_set;
} ecs_exec_velocity_setdata_t;

typedef struct { /* unpacked */
        uint32_t entity_startid8x;
        vec2f32_t velocity_set[_UINT32_IN_AVX2REG]; /* 2 simd regs */
} ecs_exec_velocity_setdata_avx2_t;

typedef struct { /* unpacked */
        uint32_t entity_id;
        vec2f32_t pos1;
        vec2f32_t pos2;
} ecs_exec_teleportdata_t;

typedef struct { /* unpacked */
        uint32_t entity_startid8x;
        vec2f32_t pos1[_UINT32_IN_AVX2REG]; /* 2 simd regs */
        vec2f32_t pos2[_UINT32_IN_AVX2REG]; /* 2 simd regs */
} ecs_exec_teleportdata_avx2_t;

typedef struct { /* unpacked */
        uint32_t entity_id;
        uint64_t state_hi_or;
        uint64_t state_lo_or;
} ecs_exec_state_ordata_t;

typedef struct { /* unpacked */
        uint32_t entity_startid8x;
        uint64_t state_hi_or[_UINT32_IN_AVX2REG]; /* 2 simd regs */
        uint64_t state_lo_or[_UINT32_IN_AVX2REG]; /* 2 simd regs */
} ecs_exec_state_ordata_avx2_t;

typedef struct { /* unpacked */
        uint32_t entity_id;
        uint64_t state_hi_and;
        uint64_t state_lo_and;
} ecs_exec_state_anddata_t;

typedef struct { /* unpacked */
        uint32_t entity_startid8x;
        uint64_t state_hi_and[_UINT32_IN_AVX2REG]; /* 2 simd regs */
        uint64_t state_lo_and[_UINT32_IN_AVX2REG]; /* 2 simd regs */
} ecs_exec_state_anddata_avx2_t;

#define ecs_exec(ptr) _Generic((ptr), \
        ecs_exec_spawndata_t*: ecs_exec_spawn, \
        ecs_exec_killdata_t*: ecs_exec_kill, \
        ecs_exec_velocity_acceldata_t*: ecs_exec_velocity_accel, \
        ecs_exec_velocity_setdata_t*: ecs_exec_velocity_set, \
        ecs_exec_teleportdata_t*: ecs_exec_teleport, \
        ecs_exec_state_ordata_t*: ecs_exec_state_or, \
        ecs_exec_state_anddata_t*: ecs_exec_state_and \
)(ptr)

#define ecs_exec_avx2(ptr) _Generic((ptr), \
        ecs_exec_spawndata_avx2_t*: ecs_exec_spawn_avx2, \
        ecs_exec_killdata_avx2_t*: ecs_exec_kill_avx2, \
        ecs_exec_velocity_acceldata_avx2_t*: ecs_exec_velocity_accel_avx2, \
        ecs_exec_velocity_setdata_avx2_t*: ecs_exec_velocity_set_avx2, \
        ecs_exec_teleportdata_avx2_t*: ecs_exec_teleport_avx2, \
        ecs_exec_state_ordata_avx2_t*: ecs_exec_state_or_avx2, \
        ecs_exec_state_anddata_avx2_t*: ecs_exec_state_and_avx2 \
)(ptr)

void ecs_exec_spawn(ecs_handle_t* ecs_handle, ecs_exec_spawndata_t* data);
void ecs_exec_kill(ecs_handle_t* ecs_handle, ecs_exec_killdata_t* data);
void ecs_exec_velocity_accel(ecs_handle_t* ecs_handle, ecs_exec_velocity_acceldata_t* data);
void ecs_exec_velocity_set(ecs_handle_t* ecs_handle, ecs_exec_velocity_setdata_t* data);
void ecs_exec_teleport(ecs_handle_t* ecs_handle, ecs_exec_teleportdata_t* data);
void ecs_exec_state_or(ecs_handle_t* ecs_handle, ecs_exec_state_ordata_t* data);
void ecs_exec_state_and(ecs_handle_t* ecs_handle, ecs_exec_state_anddata_t* data);

void ecs_exec_spawn_avx2(ecs_handle_t* ecs_handle, ecs_exec_spawndata_avx2_t* data);
void ecs_exec_kill_avx2(ecs_handle_t* ecs_handle, ecs_exec_killdata_avx2_t* data);
void ecs_exec_velocity_accel_avx2(ecs_handle_t* ecs_handle, ecs_exec_velocity_acceldata_avx2_t* data);
void ecs_exec_velocity_set_avx2(ecs_handle_t* ecs_handle, ecs_exec_velocity_setdata_avx2_t* data);
void ecs_exec_teleport_avx2(ecs_handle_t* ecs_handle, ecs_exec_teleportdata_avx2_t* data);
void ecs_exec_state_or_avx2(ecs_handle_t* ecs_handle, ecs_exec_state_ordata_avx2_t* data);
void ecs_exec_state_and_avx2(ecs_handle_t* ecs_handle, ecs_exec_state_anddata_avx2_t* data);

ecs_handle_t ecs_handled_create(void);
void ecs_handled_destroy(ecs_handle_t* ecs_handle);
void ecs_update(ecs_handle_t* ecs_handle, float32_t dt);
void ecs_zero_out(ecs_handle_t* ecs_handle);
ecs_shared_t* ecs_shared_devbuf_create(void);
void ecs_shared_devbuf_destroy(ecs_shared_t* devbuf);
vec2f32_t* ecs_pos1_devbuf_create(void);
void ecs_pos1_devbuf_destroy(vec2f32_t* devbuf);
#endif
