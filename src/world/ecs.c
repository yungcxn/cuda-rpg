
#include <stdint.h>
#include <immintrin.h>
#include "../def.h"
#include "../types/vec.h"
#include "../render/tileinfo.h"
#include "ecs.h"

uint32_t ecs_entity_count;
uint32_t ecs_killed_count;
ecs_t* ecs_instance;

#define X_ENTITY_TUPLE(name_id, updatefunc) updatefunc,
static const ecs_entity_updatefunc_t ecs_entity_updatefunc_table[] = {
    ECS_ENTITYLIST
};
#undef X_ENTITY_TUPLE

void ecs_exec_spawn(ecs_exec_spawndata_t* restrict data) {
        if (ecs_entity_count >= ECS_MAX_ENTITIES) THROW("ECS entity limit reached");
        uint32_t entity_id = ecs_entity_count++;
        ecs_instance->state_hi[entity_id] = 0;
        ecs_instance->state_lo[entity_id] = 0;
        ecs_instance->pos1[entity_id] = data->pos1;
        ecs_instance->pos2[entity_id] = data->pos2;
        ecs_instance->velocity[entity_id] = VEC2_NULL;
        ecs_instance->entitytype[entity_id] = data->entitytype;
        ecs_instance->root_entity_id[entity_id] = data->root_entity_id;
        ecs_instance->spriteinfo[entity_id] = data->spriteinfo;
}

void ecs_exec_spawn_avx2(ecs_exec_spawndata_avx2_t* restrict data) {
        if (ecs_entity_count + _UINT32_IN_AVX2REG > ECS_MAX_ENTITIES) THROW("ECS entity limit reached");
        uint32_t aligned_entitystartid = ecs_entity_count;
        if (aligned_entitystartid % _UINT32_IN_AVX2REG != 0) THROW("ECS entity start ID not aligned to AVX2 register size");
        ecs_entity_count += _UINT32_IN_AVX2REG;
        
        __m256 pos1_h1 = _mm256_load_ps((float*)&data->pos1[0]);
        __m256 pos1_h2 = _mm256_load_ps((float*)&data->pos1[4]);
        __m256 pos2_h1 = _mm256_load_ps((float*)&data->pos2[0]);
        __m256 pos2_h2 = _mm256_load_ps((float*)&data->pos2[4]);
        __m256i entitytype_vec = _mm256_load_si256((__m256i*)&data->entitytype[0]);
        __m256i root_entity_id_vec = _mm256_load_si256((__m256i*)&data->root_entity_id[0]);
        __m256i spriteinfo_vec = _mm256_load_si256((__m256i*)&data->spriteinfo[0]);
        
        __m256i state_zero = _mm256_setzero_si256();
        __m256 velocity_zero = _mm256_setzero_ps();
        
        _mm256_store_si256((__m256i*)&ecs_instance->state_hi[aligned_entitystartid], state_zero);
        _mm256_store_si256((__m256i*)&ecs_instance->state_hi[aligned_entitystartid + 4], state_zero);
        _mm256_store_si256((__m256i*)&ecs_instance->state_lo[aligned_entitystartid], state_zero);
        _mm256_store_si256((__m256i*)&ecs_instance->state_lo[aligned_entitystartid + 4], state_zero);
        _mm256_store_ps((float*)&ecs_instance->pos1[aligned_entitystartid], pos1_h1);
        _mm256_store_ps((float*)&ecs_instance->pos1[aligned_entitystartid + 4], pos1_h2);
        _mm256_store_ps((float*)&ecs_instance->pos2[aligned_entitystartid], pos2_h1);
        _mm256_store_ps((float*)&ecs_instance->pos2[aligned_entitystartid + 4], pos2_h2);
        _mm256_store_ps((float*)&ecs_instance->velocity[aligned_entitystartid], velocity_zero);
        _mm256_store_ps((float*)&ecs_instance->velocity[aligned_entitystartid + 4], velocity_zero);
        _mm256_store_si256((__m256i*)&ecs_instance->entitytype[aligned_entitystartid], entitytype_vec);
        _mm256_store_si256((__m256i*)&ecs_instance->root_entity_id[aligned_entitystartid], root_entity_id_vec);
        _mm256_store_si256((__m256i*)&ecs_instance->spriteinfo[aligned_entitystartid], spriteinfo_vec);
}

void ecs_exec_kill(ecs_exec_killdata_t* restrict data) {
        uint32_t entity_id = data->entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID to kill: %u", entity_id);
        ecs_instance->state_hi[entity_id] |= ECS_ENTITY_DEADMASK;
        ecs_killed_count++;
}

void ecs_exec_kill_avx2(ecs_exec_killdata_avx2_t* restrict data) {
        uint32_t entity_startid8x = data->entity_startid8x;
        if (entity_startid8x % _UINT32_IN_AVX2REG != 0) THROW("ECS entity start ID not aligned to AVX2 register size");
        if (entity_startid8x + _UINT32_IN_AVX2REG > ecs_entity_count) THROW("Invalid entity ID to kill AVX2: %u", entity_startid8x);
        /* patch state_hi */
        __m256i deadmask = _mm256_set1_epi64x(ECS_ENTITY_DEADMASK);
        for (uint32_t i = 0; i < 2; i++) {
                __m256i state_hi = _mm256_load_si256((__m256i*)&ecs_instance->state_hi[entity_startid8x + i * 4]);
                state_hi = _mm256_or_si256(state_hi, deadmask);
                _mm256_store_si256((__m256i*)&ecs_instance->state_hi[entity_startid8x + i * 4], state_hi);
        }
        ecs_killed_count += _UINT32_IN_AVX2REG;
}

void ecs_exec_velocity_accel(ecs_exec_velocity_acceldata_t* restrict data) {
        uint32_t entity_id = data->entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID for rotational velocity update: %u", entity_id);
        vec2f32_add(&ecs_instance->velocity[entity_id], &data->velocity_diff);
}

void ecs_exec_velocity_accel_avx2(ecs_exec_velocity_acceldata_avx2_t* restrict data) {
        uint32_t entity_startid8x = data->entity_startid8x;
        if (entity_startid8x % _UINT32_IN_AVX2REG != 0) THROW("ECS entity start ID not aligned to AVX2 register size");
        if (entity_startid8x + _UINT32_IN_AVX2REG > ecs_entity_count) THROW("Invalid entity ID for rotational velocity update AVX2: %u", entity_startid8x);
        for (uint32_t i = 0; i < 2; i++) {
                __m256 vel = _mm256_load_ps((float*)&ecs_instance->velocity[entity_startid8x + i * 4]);
                __m256 vel_diff = _mm256_load_ps((float*)&data->velocity_diff[i * 4]);
                vel = _mm256_add_ps(vel, vel_diff);
                _mm256_store_ps((float*)&ecs_instance->velocity[entity_startid8x + i * 4], vel);
        }
}

void ecs_exec_velocity_set(ecs_exec_velocity_setdata_t* restrict data) {
        uint32_t entity_id = data->entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID for rotational velocity set: %u", entity_id);
        ecs_instance->velocity[entity_id] = data->velocity_set;
}

void ecs_exec_velocity_set_avx2(ecs_exec_velocity_setdata_avx2_t* restrict data) {
        uint32_t entity_startid8x = data->entity_startid8x;
        if (entity_startid8x % _UINT32_IN_AVX2REG != 0) THROW("ECS entity start ID not aligned to AVX2 register size");
        if (entity_startid8x + _UINT32_IN_AVX2REG > ecs_entity_count) THROW("Invalid entity ID for rotational velocity set AVX2: %u", entity_startid8x);
        for (uint32_t i = 0; i < 2; i++) {
                __m256 vel_set = _mm256_load_ps((float*)&data->velocity_set[i * 4]);
                _mm256_store_ps((float*)&ecs_instance->velocity[entity_startid8x + i * 4], vel_set);
        }
}

void ecs_exec_teleport(ecs_exec_teleportdata_t* restrict data) {
        uint32_t entity_id = data->entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID for teleport: %u", entity_id);
        ecs_instance->pos1[entity_id] = data->pos1;
        ecs_instance->pos2[entity_id] = data->pos2;
}

void ecs_exec_teleport_avx2(ecs_exec_teleportdata_avx2_t* restrict data) {
        uint32_t entity_startid8x = data->entity_startid8x;
        if (entity_startid8x % _UINT32_IN_AVX2REG != 0) THROW("ECS entity start ID not aligned to AVX2 register size");
        if (entity_startid8x + _UINT32_IN_AVX2REG > ecs_entity_count) THROW("Invalid entity ID for teleport AVX2: %u", entity_startid8x);
        for (uint32_t i = 0; i < 2; i++) {
                __m256 pos1 = _mm256_load_ps((float*)&data->pos1[i * 4]);
                __m256 pos2 = _mm256_load_ps((float*)&data->pos2[i * 4]);
                _mm256_store_ps((float*)&ecs_instance->pos1[entity_startid8x + i * 4], pos1);
                _mm256_store_ps((float*)&ecs_instance->pos2[entity_startid8x + i * 4], pos2);
        }
}

void ecs_exec_state_or(ecs_exec_state_ordata_t* restrict data) {
        uint32_t entity_id = data->entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID for state OR: %u", entity_id);
        ecs_instance->state_hi[entity_id] |= data->state_hi_or;
        ecs_instance->state_lo[entity_id] |= data->state_lo_or;
}

void ecs_exec_state_or_avx2(ecs_exec_state_ordata_avx2_t* restrict data) {
        uint32_t entity_startid8x = data->entity_startid8x;
        if (entity_startid8x % _UINT32_IN_AVX2REG != 0) THROW("ECS entity start ID not aligned to AVX2 register size");
        if (entity_startid8x + _UINT32_IN_AVX2REG > ecs_entity_count) THROW("Invalid entity ID for state OR AVX2: %u", entity_startid8x);

        for (uint32_t i = 0; i < 2; i++) {
                __m256i state_hi = _mm256_load_si256((__m256i*)&ecs_instance->state_hi[entity_startid8x + i * 4]);
                __m256i state_lo = _mm256_load_si256((__m256i*)&ecs_instance->state_lo[entity_startid8x + i * 4]);
                __m256i state_hi_or = _mm256_load_si256((__m256i*)&data->state_hi_or[i * 4]);
                __m256i state_lo_or = _mm256_load_si256((__m256i*)&data->state_lo_or[i * 4]);

                state_hi = _mm256_or_si256(state_hi, state_hi_or);
                state_lo = _mm256_or_si256(state_lo, state_lo_or);

                _mm256_store_si256((__m256i*)&ecs_instance->state_hi[entity_startid8x + i * 4], state_hi);
                _mm256_store_si256((__m256i*)&ecs_instance->state_lo[entity_startid8x + i * 4], state_lo);
        }
}

void ecs_exec_state_and(ecs_exec_state_anddata_t* restrict data) {
        uint32_t entity_id = data->entity_id;
        if (entity_id >= ecs_entity_count) THROW("Invalid entity ID for state AND: %u", entity_id);
        ecs_instance->state_hi[entity_id] &= data->state_hi_and;
        ecs_instance->state_lo[entity_id] &= data->state_lo_and;
}

void ecs_exec_state_and_avx2(ecs_exec_state_anddata_avx2_t* restrict data) {
        uint32_t entity_startid8x = data->entity_startid8x;
        if (entity_startid8x % _UINT32_IN_AVX2REG != 0) THROW("ECS entity start ID not aligned to AVX2 register size");
        if (entity_startid8x + _UINT32_IN_AVX2REG > ecs_entity_count) THROW("Invalid entity ID for state AND AVX2: %u", entity_startid8x);
        for (uint32_t i = 0; i < 2; i++) {
                __m256i state_hi = _mm256_load_si256((__m256i*)&ecs_instance->state_hi[entity_startid8x + i * 4]);
                __m256i state_lo = _mm256_load_si256((__m256i*)&ecs_instance->state_lo[entity_startid8x + i * 4]);
                __m256i state_hi_and = _mm256_load_si256((__m256i*)&data->state_hi_and[i * 4]);
                __m256i state_lo_and = _mm256_load_si256((__m256i*)&data->state_lo_and[i * 4]);
                state_hi = _mm256_and_si256(state_hi, state_hi_and);
                state_lo = _mm256_and_si256(state_lo, state_lo_and);
                _mm256_store_si256((__m256i*)&ecs_instance->state_hi[entity_startid8x + i * 4], state_hi);
                _mm256_store_si256((__m256i*)&ecs_instance->state_lo[entity_startid8x + i * 4], state_lo);
        }
}

ecs_t* ecs_get() {
        if (!ecs_instance) THROW("ECS instance is not allocated");
        return ecs_instance;
}

static inline void _ecs_alloc() {
        if (ecs_instance) THROW("ECS instance already allocated");
        ecs_instance = (ecs_t*) aligned_alloc(32, sizeof(ecs_t));
        if (!ecs_instance) THROW("Failed to allocate ECS");
        memset(ecs_instance, 0, sizeof(ecs_t));
        ecs_entity_count = 0;
        ecs_killed_count = 0;
}

static inline void _free() {
        free(ecs_instance);
}

void ecs_setup() {
        _ecs_alloc();
}

void ecs_cleanup() {
        _free();
}

/* does not operate on instr cache */
static inline void _update_positions(float dt) {
        /* each AVX2 register holds 4 vec2f32_t (8 floats: x,y,x,y,x,y,x,y) */
        /*  aswell as 4 vec2f32_t (8 floats: dx, dy, dx, dy, dx, dy, dx, dy) */
        ecs_t* e = ecs_instance;
        const uint32_t n = ecs_entity_count;
        __m256 dt_vec = _mm256_set1_ps(dt);

        uint32_t i = 0;
        for (; i + 4 <= n; i += 4)
                __m256 pos1 = _mm256_load_ps((float*)&e->pos1[i]);
                __m256 pos2 = _mm256_load_ps((float*)&e->pos2[i]);
                __m256 vel  = _mm256_load_ps((float*)&e->velocity[i]);

                __m256 delta = _mm256_mul_ps(vel, dt_vec);
                _mm256_store_ps((float*)&e->pos1[i], _mm256_add_ps(pos1, delta));
                _mm256_store_ps((float*)&e->pos2[i], _mm256_add_ps(pos2, delta));
        }

        for (; i < n; i++) {
                e->pos1[i].x += e->velocity[i].x * dt;
                e->pos1[i].y += e->velocity[i].y * dt;
                e->pos2[i].x += e->velocity[i].x * dt;
                e->pos2[i].y += e->velocity[i].y * dt;
        }
}

/* again, only operates on ecs aswell */
static inline void _remove_garbage() {
        uint32_t j = 0;
        for (uint32_t i = 0; i < ecs_entity_count; i++) {
                uint64_t state_hi_i = ecs_instance->state_hi[i];
                if (!ECS_IS_ENTITY_DEAD(state_hi_i)) {
                        ecs_instance->state_hi[j] = state_hi_i;
                        ecs_instance->state_lo[j] = ecs_instance->state_lo[i];
                        ecs_instance->pos1[j] = ecs_instance->pos1[i];
                        ecs_instance->pos2[j] = ecs_instance->pos2[i];
                        ecs_instance->velocity[j] = ecs_instance->velocity[i];
                        ecs_instance->entitytype[j] = ecs_instance->entitytype[i];
                        ecs_instance->root_entity_id[j] = ecs_instance->root_entity_id[i];
                        ecs_instance->spriteinfo[j] = ecs_instance->spriteinfo[i];
                        j++;
                }
        }
        ecs_killed_count = 0;
        ecs_entity_count = j;
}


void ecs_update(float dt) {
        _update_positions(dt);
        if (ecs_killed_count > ECS_GARBAGE_REMOVAL_THRESHOLD) _remove_garbage();
        for (uint32_t i = 0; i < ecs_entity_count; i++) {
                if (ECS_IS_ENTITY_DEAD(ecs_instance->state_hi[i])) continue;
                if (ecs_instance->entitytype[i] >= ECS_ENTITYTYPE_COUNT) THROW("Invalid entity type for entity ID %u: %u", i, ecs_instance->entitytype[i]);
                ecs_entity_updatefunc_table[ecs_instance->entitytype[i]](i, dt);
        }
}

uint32_t ecs_get_entity_count() {
        return ecs_entity_count;
}

uint32_t ecs_get_killed_count() {
        return ecs_killed_count;
}