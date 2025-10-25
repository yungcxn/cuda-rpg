#ifndef ECS_H
#define ECS_H

#include <stdint.h>
#include "../types/vec.h"
#include "../render/spriteinfo.cuh"

#define ECS_MAX_ENTITIES 1024
#define ECS_ENTITY_DEADMASK 0x8000000000000000ULL
#define ECS_IS_ENTITY_DEAD(state) ((state) & ECS_ENTITY_DEADMASK)
#define ECS_CODENAME_SIZE 4  /* 32-bit codename */
#define ECS_INSTR_CACHE_SIZE 1024  /* number of instructions in cache */
#define ECS_GARBAGE_REMOVAL_THRESHOLD 100  /* number of killed entities to trigger garbage removal */

typedef struct __attribute__((packed)) {
        uint64_t state[ECS_MAX_ENTITIES]; /* 64 */
        vec2f32_t pos1[ECS_MAX_ENTITIES]; /* + 64 => 128 */
        vec2f32_t pos2[ECS_MAX_ENTITIES];   /* + 64 => 192 */
        vec2f32_t velocity[ECS_MAX_ENTITIES];  /* + 64 => 256 */
        spriteinfo_id_t tile_id[ECS_MAX_ENTITIES]; /* + 32 => 288 */
        char codename[ECS_CODENAME_SIZE * ECS_MAX_ENTITIES]; /* + 32 => 320 */
} ecs_t;

typedef enum {
        ECS_INSTR_NOP = 0,
        ECS_INSTR_SPAWN = 1,
        ECS_INSTR_KILL = 2,
        ECS_INSTR_VELOCITY_ACCEL = 3,
        ECS_INSTR_VELOCITY_SET = 4,
        ECS_INSTR_TELEPORT = 5,
        ECS_INSTR_STATE_OR = 6,
        ECS_INSTR_STATE_AND = 7,
} ecs_instr_opcode_t;

typedef struct __attribute__((packed, aligned(32))) {
        uint32_t opcode; /* 32 */
        union {
                struct {
                        uint32_t nothing[7]; /* + 224 => 256 */
                } nop;
                struct {
                        char codename[ECS_CODENAME_SIZE]; /* + 32 => 64 */
                        vec2f32_t pos1; /* + 64 => 128 */
                        vec2f32_t pos2; /* + 64 => 192 */
                        vec2f32_t velocity; /* + 64 => 256 */
                } spawn;
                struct {
                        uint32_t entity_id; /* + 32 => 64 */
                } kill;
                struct {
                        uint32_t entity_id;  /* + 32 => 64 */
                        vec2f32_t velocity_diff; /* + 64 => 128 */
                } velocity_accel;
                struct {
                        uint32_t entity_id;  /* + 32 => 64 */
                        vec2f32_t velocity_set; /* + 64 => 128 */
                } velocity_set;
                struct {
                        uint32_t entity_id; /* + 32 => 64 */
                        vec2f32_t pos1; /* + 64 => 128 */
                        vec2f32_t pos2; /* + 64 => 192 */
                } teleport;
                struct {
                        uint32_t entity_id; /* + 32 => 64 */
                        uint64_t state_or; /* + 64 => 128 */
                } state_or;
                struct {
                        uint32_t entity_id; /* + 32 => 64 */
                        uint64_t state_and; /* + 64 => 128 */
                } state_and;
        };
} ecs_instr_t; /* must be 256-bit for AVX2 */

#define ECS_SPAWN_INSTR(codename_, pos1_, pos2_, velocity_) \
        ((ecs_instr_t){ \
                .opcode = ECS_INSTR_SPAWN, \
                .spawn = { \
                        .codename = codename_, \
                        .pos1 = pos1_, \
                        .pos2 = pos2_, \
                        .velocity = velocity_ \
                } \
        })

#define ECS_KILL_INSTR(entity_id_) \
        ((ecs_instr_t){ \
                .opcode = ECS_INSTR_KILL, \
                .kill = { \
                        .entity_id = entity_id_ \
                } \
        })

#define ECS_VELOCITY_ACCEL_INSTR(entity_id_, velocity_diff_) \
        ((ecs_instr_t){ \
                .opcode = ECS_INSTR_VELOCITY_UPDATE, \
                .velocity_accel = { \
                        .entity_id = entity_id_, \
                        .velocity_diff = velocity_diff_ \
                } \
        })

#define ECS_TELEPORT_INSTR(entity_id_, pos1_, pos2_) \
        ((ecs_instr_t){ \
                .opcode = ECS_INSTR_TELEPORT, \
                .teleport = { \
                        .entity_id = entity_id_, \
                        .pos1 = pos1_, \
                        .pos2 = pos2_ \
                } \
        })

#define ECS_STATE_OR_INSTR(entity_id_, state_or_) \
        ((ecs_instr_t){ \
                .opcode = ECS_INSTR_STATE_OR, \
                .state_or = { \
                        .entity_id = entity_id_, \
                        .state_or = state_or_ \
                } \
        })

#define ECS_STATE_AND_INSTR(entity_id_, state_and_) \
        ((ecs_instr_t){ \
                .opcode = ECS_INSTR_STATE_AND, \
                .state_and = { \
                        .entity_id = entity_id_, \
                        .state_and = state_and_ \
                } \
        })

#define ECS_VELOCITY_SET_INSTR(entity_id_, velocity_set_) \
        ((ecs_instr_t){ \
                .opcode = ECS_INSTR_VELOCITY_SET, \
                .velocity_set = { \
                        .entity_id = entity_id_, \
                        .velocity_set = velocity_set_ \
                } \
        })

ecs_t* ecs_get();
void ecs_setup();
void ecs_cleanup();
void ecs_update(float dt);
void ecs_queue_exec(ecs_instr_t instr);
void ecs_queue_execs(ecs_instr_t* instr, uint32_t instr_count);

#endif