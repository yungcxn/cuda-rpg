#ifndef ECS_H
#define ECS_H

#include <stdint.h>
#include "../types/vec.h"
#include "../render/spriteinfo.h"

#define ECS_MAX_ENTITIES 1024
#define ECS_ENTITY_DEADMASK 0x8000000000000000ULL
#define ECS_IS_ENTITY_DEAD(state_hi) ((state_hi) & ECS_ENTITY_DEADMASK)
#define ECS_GARBAGE_REMOVAL_THRESHOLD 100  /* number of killed entities to trigger garbage removal */
#define _UINT32_IN_AVX2REG 8

#define ECS_ENTITYLIST \
        X_ENTITYTYPE_TUPLE(ECS_ENTITY_PLAYER, entityplayer_update) \
        X_ENTITYTYPE_TUPLE(ECS_ENTITY_ZOMBIE, entityzombie_update)

#define X_ENTITYTYPE_TUPLE(name_id, updatefunc) name_id,
typedef enum {
        ECS_ENTITYLIST
        ECS_ENTITYTYPE_COUNT
} ecs_entitytype_t;
#undef X_ENTITYTYPE_TUPLE
_Static_assert(sizeof(ecs_entitytype_t) == 4, "ecs_entitytype_t must be 32-bit");

typedef void (*ecs_entity_updatefunc_t)(uint32_t entity_id, float32_t dt);

typedef struct __attribute__((aligned(32))) {
        uint64_t state_hi[ECS_MAX_ENTITIES]; /* 64 */
        uint64_t state_lo[ECS_MAX_ENTITIES]; /* + 64 => 128 */
        vec2f32_t pos1[ECS_MAX_ENTITIES]; /* + 64 => 192 */
        vec2f32_t pos2[ECS_MAX_ENTITIES]; /* + 64 => 256 */
        vec2f32_t velocity[ECS_MAX_ENTITIES]; /* + 64 => 320 */
        /* we handle sprites on GPU */
        ecs_entitytype_t entitytype[ECS_MAX_ENTITIES]; /* + 32 => 352 */
        uint32_t root_entity_id[ECS_MAX_ENTITIES]; /* + 32 => 384 */
        spriteinfo_id_t spriteinfo[ECS_MAX_ENTITIES]; /* + 32 => 400 */
} ecs_t;

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

void ecs_exec_spawn(ecs_exec_spawndata_t* data);
void ecs_exec_kill(ecs_exec_killdata_t* data);
void ecs_exec_velocity_accel(ecs_exec_velocity_acceldata_t* data);
void ecs_exec_velocity_set(ecs_exec_velocity_setdata_t* data);
void ecs_exec_teleport(ecs_exec_teleportdata_t* data);
void ecs_exec_state_or(ecs_exec_state_ordata_t* data);
void ecs_exec_state_and(ecs_exec_state_anddata_t* data);

void ecs_exec_spawn_avx2(ecs_exec_spawndata_avx2_t* data);
void ecs_exec_kill_avx2(ecs_exec_killdata_avx2_t* data);
void ecs_exec_velocity_accel_avx2(ecs_exec_velocity_acceldata_avx2_t* data);
void ecs_exec_velocity_set_avx2(ecs_exec_velocity_setdata_avx2_t* data);
void ecs_exec_teleport_avx2(ecs_exec_teleportdata_avx2_t* data);
void ecs_exec_state_or_avx2(ecs_exec_state_ordata_avx2_t* data);
void ecs_exec_state_and_avx2(ecs_exec_state_anddata_avx2_t* data);

ecs_t* ecs_get();
void ecs_setup();
void ecs_cleanup();
void ecs_update(float dt);

uint32_t ecs_get_entity_count();
uint32_t ecs_get_killed_count();

#endif
