
#include <stdint.h>
#include <immintrin.h>
#include "../def.h"
#include "../types/vec.h"
#include "../render/tileinfo.cuh"
#include "ecs.h"

typedef void (*ecs_op_func_t)(ecs_instr_t* instr);

ecs_instr_t* ecs_instr_cache;
uint32_t instr_hp; /* head pointer */
uint32_t ecs_entity_count;
uint32_t ecs_killed_count;
ecs_t* ecs_instance;

[[maybe_unused]] static void _instr_nop([[maybe_unused]] ecs_instr_t* instr) { THROW("NOP instruction should not run"); }

static void _instr_spawn(ecs_instr_t* instr) {
        if (ecs_entity_count >= ECS_MAX_ENTITIES) THROW("ECS entity limit reached");
        uint32_t entity_id = ecs_entity_count++;
        ecs_instance->state[entity_id] = 0;
        /* set codename */
        memmove(&ecs_instance->codename[ECS_CODENAME_SIZE * entity_id],
                instr->spawn.codename,
                ECS_CODENAME_SIZE);
        /* set positions */
        ecs_instance->pos1[entity_id] = instr->spawn.pos1;
        ecs_instance->pos2[entity_id] = instr->spawn.pos2;
        /* set rotational velocity */
        ecs_instance->velocity[entity_id] = instr->spawn.velocity;
}

static void _instr_kill(ecs_instr_t* instr) {
        uint32_t entity_id = instr->kill.entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID to kill: %u", entity_id);
        ecs_instance->state[entity_id] |= ECS_ENTITY_DEADMASK;
        ecs_killed_count++;
}

static void _instr_velocity_accel(ecs_instr_t* instr) {
        uint32_t entity_id = instr->velocity_accel.entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID for rotational velocity update: %u", entity_id);
        vec2f32_add(&ecs_instance->velocity[entity_id], &instr->velocity_accel.velocity_diff);
}

static void _instr_velocity_set(ecs_instr_t* instr) {
        uint32_t entity_id = instr->velocity_set.entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID for rotational velocity set: %u", entity_id);
        ecs_instance->velocity[entity_id] = instr->velocity_set.velocity_set;
}

static void _instr_teleport(ecs_instr_t* instr) {
        uint32_t entity_id = instr->teleport.entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID for teleport: %u", entity_id);
        ecs_instance->pos1[entity_id] = instr->teleport.pos1;
        ecs_instance->pos2[entity_id] = instr->teleport.pos2;
}

static void _instr_state_or(ecs_instr_t* instr) {
        uint32_t entity_id = instr->state_or.entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID for state OR: %u", entity_id);
        ecs_instance->state[entity_id] |= instr->state_or.state_or;
}

static void _instr_state_and(ecs_instr_t* instr) {
        uint32_t entity_id = instr->state_and.entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID for state AND: %u", entity_id);
        ecs_instance->state[entity_id] &= instr->state_and.state_and;
}

ecs_op_func_t ecs_op_funcs[] = {
        [ECS_INSTR_NOP] = _instr_nop,
        [ECS_INSTR_SPAWN] = _instr_spawn,
        [ECS_INSTR_KILL] = _instr_kill,
        [ECS_INSTR_VELOCITY_ACCEL] = _instr_velocity_accel,
        [ECS_INSTR_STATE_OR] = _instr_state_or,
        [ECS_INSTR_STATE_AND] = _instr_state_and,
        [ECS_INSTR_TELEPORT] = _instr_teleport,
        [ECS_INSTR_VELOCITY_SET] = _instr_velocity_set,
};

ecs_t* ecs_get() {
        if (!ecs_instance) THROW("ECS instance is not allocated");
        return ecs_instance;
}

static inline void _check_instr_cache_type_constraints() {
        if (sizeof(ecs_instr_t) != sizeof(__m256i)) THROW("ecs_instr_t must be 256-bit for AVX2 processing");
}

static inline void _alloc() {
        ecs_instance = (ecs_t*) aligned_alloc(32, sizeof(ecs_t));
        if (!ecs_instance) THROW("Failed to allocate ECS");
        memset(ecs_instance, 0, sizeof(ecs_t));
        ecs_entity_count = 0;
        ecs_killed_count = 0;
}



static inline void _instr_cache_alloc() {
        _check_instr_cache_type_constraints();
        ecs_instr_cache = (ecs_instr_t*) aligned_alloc(32, sizeof(ecs_instr_t) * ECS_INSTR_CACHE_SIZE);
        instr_hp = 0;
        if (!ecs_instr_cache) THROW("Failed to allocate ECS instruction cache");
        memset(ecs_instr_cache, 0, sizeof(ecs_instr_t) * ECS_INSTR_CACHE_SIZE);
}

static inline void _free() {
        free(ecs_instance);
}

static inline void _instr_cache_free() {
        free(ecs_instr_cache);
}

void ecs_setup() {
        _alloc();
        _instr_cache_alloc();
}

void ecs_cleanup() {
        _instr_cache_free();
        _free();
}

static inline void _update_positions(float dt) {
    /* each AVX2 register holds 4 vec2f32_t (8 floats: x,y,x,y,x,y,x,y) */
    /*  aswell as 4 vec2f32_t (8 floats: dx, dy, dx, dy, dx, dy, dx, dy) */
    uint32_t avx_batches = ecs_entity_count / 4;
    __m256 dt_vec = _mm256_set1_ps(dt);

    for (uint32_t i = 0; i < avx_batches; i++) {
        uint32_t base = i * 4;
        __m256 pos1 = _mm256_load_ps((float*)&ecs_instance->pos1[base]);
        __m256 velocity = _mm256_load_ps((float*)&ecs_instance->velocity[base]);

        /* compute delta = velocity * dt */
        __m256 dt_vec = _mm256_set1_ps(dt);
        __m256 delta = _mm256_mul_ps(velocity, dt_vec);

        /* update pos1 += delta */
        __m256 new_pos1 = _mm256_add_ps(pos1, delta);
        _mm256_store_ps((float*)&ecs_instance->pos1[base], new_pos1);

        /* update pos2 += delta */
        __m256 pos2 = _mm256_load_ps((float*)&ecs_instance->pos2[base]);
        __m256 new_pos2 = _mm256_add_ps(pos2, delta);
        _mm256_store_ps((float*)&ecs_instance->pos2[base], new_pos2);
    }
    
    /* remainder entities */
    for (uint32_t i = avx_batches * 4; i < ecs_entity_count; i++) {
        ecs_instance->pos1[i].x += ecs_instance->velocity[i].x * dt;
        ecs_instance->pos1[i].y += ecs_instance->velocity[i].y * dt;
        ecs_instance->pos2[i].x += ecs_instance->velocity[i].x * dt;
        ecs_instance->pos2[i].y += ecs_instance->velocity[i].y * dt;
    }
}

static inline void _remove_garbage() {
        uint32_t j = 0;
        for (uint32_t i = 0; i < ecs_entity_count; i++) {
                uint64_t state_i = ecs_instance->state[i];
                if (!ECS_IS_ENTITY_DEAD(state_i)) {
                        ecs_instance->state[j] = state_i;
                        ecs_instance->pos1[j] = ecs_instance->pos1[i];
                        ecs_instance->pos2[j] = ecs_instance->pos2[i];
                        ecs_instance->velocity[j] = ecs_instance->velocity[i];
                        ecs_instance->tile_id[j] = ecs_instance->tile_id[i];
                        if (i != j) {
                                memmove(&ecs_instance->codename[ECS_CODENAME_SIZE * j],
                                        &ecs_instance->codename[ECS_CODENAME_SIZE * i],
                                        ECS_CODENAME_SIZE);
                        }
                        j++;
                }
        }
        ecs_killed_count = ecs_entity_count - j;
        ecs_entity_count = j;
}

static inline void _run_instr_cache() {
        for (uint32_t i = 0; i < instr_hp; i++) {
                ecs_instr_t* instr = &ecs_instr_cache[i];
                ecs_op_func_t func = ecs_op_funcs[instr->opcode];
                if (!func) THROW("Invalid ECS instruction opcode: %u", instr->opcode);
                func(instr);
        }
        instr_hp = 0;
}


void ecs_update(float dt) {
        _update_positions(dt);
        _run_instr_cache();
        if (ecs_killed_count > ECS_GARBAGE_REMOVAL_THRESHOLD) _remove_garbage();
}

void ecs_queue_exec(ecs_instr_t instr) {
        if (instr_hp >= ECS_INSTR_CACHE_SIZE) THROW("ECS instruction cache overflow");
        ecs_instr_cache[instr_hp++] = instr;
}

void ecs_queue_execs(ecs_instr_t* instr, uint32_t instr_count) {
        if (instr_hp + instr_count > ECS_INSTR_CACHE_SIZE) THROW("ECS instruction cache overflow");
        memcpy(&ecs_instr_cache[instr_hp], instr, sizeof(ecs_instr_t) * instr_count);
        instr_hp += instr_count;
}