#ifndef ECS_H
#define ECS_H

#include <stdint.h>
#include "../headeronly/vec.h"
#include "bitfsm.h"
#include "../render/spriteinfo.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ECS_MAX_ENTITIES 1024
#define ECS_ENTITY_DEADMASK BIT64(63)

#define ECS_ISDEAD(pos1) ((pos1).x == -1.0f && (pos1).y == -1.0f)
#define ECS_SETDEAD(pos1) do { (pos1).x = -1.0f; (pos1).y = -1.0f; } while(0)
#define ECS_GARBAGE_REMOVAL_THRESHOLD 100  /* number of killed entities to trigger garbage removal*/
#define _UINT32_IN_AVX2REG 8

#define EID(name_id) _ECS_ENTITYTYPE_ID_##name_id

#define ECS_ENTITYTYPE_LIST \
        X(EID(ZOMBIE), entityzombie_update)


#define X(name_id, updatefunc) name_id,
typedef enum {
        ECS_ENTITYTYPE_LIST
        ECS_ENTITYTYPE_COUNT
} ecs_entitytype_t;
#undef X

typedef void (*ecs_entity_updatefunc_t)(uint32_t entity_id, float32_t dt);

typedef struct __attribute__((aligned(32))) {
        spriteinfo_id_t spriteinfos[ECS_MAX_ENTITIES] __attribute__((aligned(32))); /* + 32 => 400*/
        float32_t spritetimers[ECS_MAX_ENTITIES] __attribute__((aligned(32))); /* + 32 => 432 */
} ecs_shared_t; /* this is shared between dev and host */

typedef struct __attribute__((aligned(32))) {
        bitfsm_state_t state[ECS_MAX_ENTITIES] __attribute__((aligned(32))); /* 64 */
        vec2f32_t pos1[ECS_MAX_ENTITIES] __attribute__((aligned(32))); /* + 64 => 128 */
        vec2f32_t pos2[ECS_MAX_ENTITIES] __attribute__((aligned(32))); /* + 64 => 192 */
        vec2f32_t velocity[ECS_MAX_ENTITIES] __attribute__((aligned(32))); /* + 64 => 256 */
        /* we handle sprites on GPU */
        ecs_entitytype_t entitytype[ECS_MAX_ENTITIES] __attribute__((aligned(32))); /* + 32 => 288 */
        uint32_t root_entity_id[ECS_MAX_ENTITIES] __attribute__((aligned(32))); /* + 32 => 320 */
        ecs_shared_t* shared; /* + 64 => 384 */
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

typedef struct __attribute__((aligned(32))) { /* unpacked */
        vec2f32_t pos1[_UINT32_IN_AVX2REG]; /* 2 simd regs */
        vec2f32_t pos2[_UINT32_IN_AVX2REG]; /* 2 simd regs */
        ecs_entitytype_t entitytype[_UINT32_IN_AVX2REG]; /* 1 simd reg */
        uint32_t root_entity_id[_UINT32_IN_AVX2REG]; /* 1 simd reg */
        spriteinfo_id_t spriteinfo[_UINT32_IN_AVX2REG]; /* 1 simd reg */
} ecs_exec_spawndata_avx2_t;

typedef struct { /* unpacked */
        uint32_t entity_id;
} ecs_exec_killdata_t;

typedef struct __attribute__((aligned(32))) { /* unpacked */
        uint32_t entity_startid8x;
} ecs_exec_killdata_avx2_t;

typedef struct { /* unpacked */
        uint32_t entity_id;
        vec2f32_t velocity_diff;
} ecs_exec_velocity_acceldata_t;

typedef struct __attribute__((aligned(32))) { /* unpacked */
        uint32_t entity_startid8x;
        vec2f32_t velocity_diff[_UINT32_IN_AVX2REG]; /* 2 simd regs */
} ecs_exec_velocity_acceldata_avx2_t;

typedef struct { /* unpacked */
        uint32_t entity_id;
        vec2f32_t velocity_set;
} ecs_exec_velocity_setdata_t;

typedef struct __attribute__((aligned(32))) { /* unpacked */
        uint32_t entity_startid8x;
        vec2f32_t velocity_set[_UINT32_IN_AVX2REG]; /* 2 simd regs */
} ecs_exec_velocity_setdata_avx2_t;

typedef struct { /* unpacked */
        uint32_t entity_id;
        vec2f32_t pos1;
        vec2f32_t pos2;
} ecs_exec_teleportdata_t;

typedef struct __attribute__((aligned(32))) { /* unpacked */
        uint32_t entity_startid8x;
        vec2f32_t pos1[_UINT32_IN_AVX2REG]; /* 2 simd regs */
        vec2f32_t pos2[_UINT32_IN_AVX2REG]; /* 2 simd regs */
} ecs_exec_teleportdata_avx2_t;

#define ecs_exec(ptr) _Generic((ptr), \
        ecs_exec_spawndata_t*: ecs_exec_spawn, \
        ecs_exec_killdata_t*: ecs_exec_kill, \
        ecs_exec_velocity_acceldata_t*: ecs_exec_velocity_accel, \
        ecs_exec_velocity_setdata_t*: ecs_exec_velocity_set, \
        ecs_exec_teleportdata_t*: ecs_exec_teleport, \
)(ptr)

#define ecs_exec_avx2(ptr) _Generic((ptr), \
        ecs_exec_spawndata_avx2_t*: ecs_exec_spawn_avx2, \
        ecs_exec_killdata_avx2_t*: ecs_exec_kill_avx2, \
        ecs_exec_velocity_acceldata_avx2_t*: ecs_exec_velocity_accel_avx2, \
        ecs_exec_velocity_setdata_avx2_t*: ecs_exec_velocity_set_avx2, \
        ecs_exec_teleportdata_avx2_t*: ecs_exec_teleport_avx2, \
)(ptr)

void ecs_exec_spawn(ecs_handle_t* ecs_handle, ecs_exec_spawndata_t* data);
void ecs_exec_kill(ecs_handle_t* ecs_handle, ecs_exec_killdata_t* data);
void ecs_exec_velocity_accel(ecs_handle_t* ecs_handle, ecs_exec_velocity_acceldata_t* data);
void ecs_exec_velocity_set(ecs_handle_t* ecs_handle, ecs_exec_velocity_setdata_t* data);
void ecs_exec_teleport(ecs_handle_t* ecs_handle, ecs_exec_teleportdata_t* data);

void ecs_exec_spawn_avx2(ecs_handle_t* ecs_handle, ecs_exec_spawndata_avx2_t* data);
void ecs_exec_kill_avx2(ecs_handle_t* ecs_handle, ecs_exec_killdata_avx2_t* data);
void ecs_exec_velocity_accel_avx2(ecs_handle_t* ecs_handle, 
                                  ecs_exec_velocity_acceldata_avx2_t* data);
void ecs_exec_velocity_set_avx2(ecs_handle_t* ecs_handle, ecs_exec_velocity_setdata_avx2_t* data);
void ecs_exec_teleport_avx2(ecs_handle_t* ecs_handle, ecs_exec_teleportdata_avx2_t* data);

ecs_handle_t ecs_handled_create(void);
void ecs_handled_destroy(ecs_handle_t* ecs_handle);
void ecs_update(ecs_handle_t* ecs_handle, float32_t dt);
void ecs_zero_out(ecs_handle_t* ecs_handle);
ecs_shared_t* ecs_shared_devbuf_create(void);
void ecs_shared_devbuf_destroy(ecs_shared_t* devbuf);
vec2f32_t* ecs_pos1_devbuf_create(void);
void ecs_pos1_devbuf_destroy(vec2f32_t* devbuf);

#ifdef __cplusplus
}
#endif

#endif
