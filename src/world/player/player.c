#include <stdlib.h>
#include "player.h"
#include "../../headeronly/def.h"
#include "../../render/util/ccuda.h"
#include "../bitfsm.h"

#define PLAYER_BB_WIDTH 1
#define PLAYER_BB_HEIGHT 1

#define PLAYER_RUNSPEED 2.0f
#define ONE_MINUS_INVSQRT2 0.29289322f

#define PLAYER_WALKING(state) BITFSM_ISSET(state, ST(WALK))
#define PLAYER_ROLLING(state) BITFSM_ISSET(state, ST(ROLL))
#define PLAYER_MOVING(state)  BITFSM_ANY(state, ST(MOVING))

player_t* player_create(void) {
        player_t* p = (player_t*)malloc(sizeof(player_t));
        ccuda_mallochost((void**)&(p->shared), sizeof(player_shared_t));
        if (!p) THROW("Failed to allocate memory for player");

        p->state = ST(MOVABLE);
        p->pos1 = (vec2f32_t){3.0f, 3.0f};
        p->pos2 = (vec2f32_t){PLAYER_BB_WIDTH, PLAYER_BB_HEIGHT};
        p->velocity = (vec2f32_t){0.0f, 0.0f};
        p->rolltimer = 0.0f;
        p->shared->spriteinfo = SID(PLAYER_IDLE_D_0);
        p->shared->spritetimer = 0.0f;

        return p;
}

void player_destroy(player_t* player) {
        if (!player) THROW("No player to destroy");
        ccuda_freehost(player->shared);
        free(player);
}

player_shared_t* player_shared_devbuf_create(void) {
        player_shared_t* devbuf;
        ccuda_malloc((void**)&devbuf, sizeof(player_shared_t));
        return devbuf;
}

void player_shared_devbuf_destroy(player_shared_t* devbuf) {
        ccuda_free(devbuf);
        devbuf = 0;
}

static inline void _update_sprite(player_t* player, spriteinfo_id_t new_sprite) {
        if (!player) THROW("No player to update sprite for");
        player->shared->spriteinfo = new_sprite;
        player->shared->spritetimer = 0.0f;
}

void player_setpos(player_t* player, vec2f32_t pos1) {
        if (!player) THROW("No player to set position for");
        player->pos1 = pos1;
        player->pos2 = (vec2f32_t){pos1.x + PLAYER_BB_WIDTH, pos1.y + PLAYER_BB_HEIGHT};
}

static inline void _check_role_end(player_t* player, float32_t dt) {
        if (!player) THROW("No player to decrement rolltimer for");
        if (player->rolltimer > 0.0f) {
                player->rolltimer -= dt;
                if (player->rolltimer <= 0.0f) {
                        player->rolltimer = 0.0f;

                        bitfsm_tryexec_cb(&player->state, TOK(END_ROLL),
                                          player_bitfsm_callback);

                }
        }
}

static inline void _apply_velocity(player_t* player, float32_t dt) {
        player->pos1.x += player->velocity.x * dt;
        player->pos1.y += player->velocity.y * dt;
        player->pos2.x = player->pos1.x + PLAYER_BB_WIDTH;
        player->pos2.y = player->pos1.y + PLAYER_BB_HEIGHT;
}

static inline void _calc_velocity(player_t* player, float32_t dt) {
        if (!player) THROW("No player to calculate velocity for");
        const bitfsm_state_t state = player->state;
        if (!PLAYER_MOVING(state)) {
                player->velocity.x = 0.0f;
                player->velocity.y = 0.0f;
                return;
        }

        int32_t x_dir;
        int32_t y_dir;
        {
                const bitfsm_state_t D = BITFSM_ISSET(state, ST(LOOK_DOWN));
                const bitfsm_state_t L = BITFSM_ISSET(state, ST(LOOK_LEFT));
                const bitfsm_state_t U = BITFSM_ISSET(state, ST(LOOK_UP));
                const bitfsm_state_t R = BITFSM_ISSET(state, ST(LOOK_RIGHT));
                x_dir = !!R - !!L;
                y_dir = !!D - !!U;
        }

        {
                const float32_t diag_mask = (float32_t)((x_dir != 0) & (y_dir != 0));
                const float32_t scale = 1.0f - diag_mask * ONE_MINUS_INVSQRT2;
                player->velocity.x = x_dir * PLAYER_RUNSPEED * scale;
                player->velocity.y = y_dir * PLAYER_RUNSPEED * scale;
        }
}

void player_update(player_t* player, float32_t dt) {
        if (!player) THROW("No player to update");

        printf("Player pos: (%.2f, %.2f)\n", player->pos1.x, player->pos1.y);
        
        /* Update position based on velocity and delta time */
        _check_role_end(player, dt);
        _calc_velocity(player, dt);
        _apply_velocity(player, dt);
}

void player_bitfsm_callback(void* playerstate, bitfsm_token_t tok) {
        player_t* player = CONTAINER_OF(playerstate, player_t, state);
        if (!player) THROW("No player in bitfsm callback");
        printf("Player FSM state changed to 0x%016llx on token %u\n", 
               *(bitfsm_state_t*)playerstate, tok);

        player_t* p = CONTAINER_OF(playerstate, player_t, state);
        bitfsm_state_t state = *(bitfsm_state_t*)playerstate;

        if (tok == TOK(ROLL)) {
                p->rolltimer = 0.5f; /* e.g., half a second roll */
        }

        if (BITFSM_ISSET(state, ST(LOOK_UP))) {
                if (PLAYER_ROLLING(state)) _update_sprite(p, SID(PLAYER_ROLL_U_0));
                else if (PLAYER_MOVING(state)) _update_sprite(p, SID(PLAYER_RUN_U_0));
                else _update_sprite(p, SID(PLAYER_IDLE_U));
        } else if (BITFSM_ISSET(state, ST(LOOK_DOWN))) {
                if (PLAYER_ROLLING(state)) _update_sprite(p, SID(PLAYER_ROLL_D_0));
                else if (PLAYER_MOVING(state)) _update_sprite(p, SID(PLAYER_RUN_D_0));
                else _update_sprite(p, SID(PLAYER_IDLE_D_0));
        } else if (BITFSM_ISSET(state, ST(LOOK_LEFT))) {
                if (PLAYER_ROLLING(state)) _update_sprite(p, SID(PLAYER_ROLL_L_0));
                else if (PLAYER_MOVING(state)) _update_sprite(p, SID(PLAYER_RUN_L_0));
                else _update_sprite(p, SID(PLAYER_IDLE_L));
        } else if (BITFSM_ISSET(state, ST(LOOK_RIGHT))) {
                if (PLAYER_ROLLING(state)) _update_sprite(p, SID(PLAYER_ROLL_R_0));
                else if (PLAYER_MOVING(state)) _update_sprite(p, SID(PLAYER_RUN_R_0));
                else _update_sprite(p, SID(PLAYER_IDLE_R));
        }
}
