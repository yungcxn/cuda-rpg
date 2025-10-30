#include <stdlib.h>
#include "player.h"
#include "../../def.h"

#define PLAYER_BB_WIDTH 1
#define PLAYER_BB_HEIGHT 1

#define PLAYER_RUNSPEED 2.0f
#define ONE_MINUS_INVSQRT2 0.29289322f
#define PLAYER_RUNSPEED_ 1.41421356f
#define STATE_LO_RUNNING_UP    BIT64(KEY_INPUT_W)
#define STATE_LO_RUNNING_DOWN  BIT64(KEY_INPUT_S)
#define STATE_LO_RUNNING_LEFT  BIT64(KEY_INPUT_A)
#define STATE_LO_RUNNING_RIGHT BIT64(KEY_INPUT_D)
#ifndef STATE_HI_DEAD
        #define STATE_HI_DEAD          BIT64(63)
#endif

player_t* player_init(void) {
        player_t* p = (player_t*)malloc(sizeof(player_t));
        if (!p) THROW("Failed to allocate memory for player");
        
        p->state_hi = 0;
        p->state_lo = 0;
        p->pos1 = (vec2f32_t){0.0f, 0.0f};
        p->pos2 = (vec2f32_t){PLAYER_BB_WIDTH, PLAYER_BB_HEIGHT};
        p->velocity = (vec2f32_t){0.0f, 0.0f};
        p->spriteinfo = SPRITEINFO_ID_PLAYER_IDLE_U;

        return p;
}

void player_destroy(player_t* player) {
        if (!player) THROW("No player to destroy");
        free(player);
}

void player_setpos(player_t* player, vec2f32_t pos1, key_inputfield_t pressed_keys) {
        if (!player) THROW("No player to set position for");
        player->pos1 = pos1;
        player->pos2 = (vec2f32_t){pos1.x + PLAYER_BB_WIDTH, pos1.y + PLAYER_BB_HEIGHT};
}

static inline void _apply_velocity(player_t* player, float32_t dt) {
        player->pos1.x += player->velocity.x * dt;
        player->pos1.y += player->velocity.y * dt;
        player->pos2.x = player->pos1.x + PLAYER_BB_WIDTH;
        player->pos2.y = player->pos1.y + PLAYER_BB_HEIGHT;
}

void player_update(player_t* player, float32_t dt, key_inputfield_t pressed_keys) {
        if (!player) THROW("No player to update");

        key_inputfield_t D = KEY_ON(pressed_keys, KEY_INPUT_D);
        key_inputfield_t A = KEY_ON(pressed_keys, KEY_INPUT_A);
        key_inputfield_t W = KEY_ON(pressed_keys, KEY_INPUT_W);
        key_inputfield_t S = KEY_ON(pressed_keys, KEY_INPUT_S);
        /* masks correspond to player->state_lo */
        player->state_lo &= (W | A | S | D);

        const uint32_t x_dir = !!D - !!A;
        const uint32_t y_dir = !!S - !!W;
        const float32_t diag_mask = (float)((x_dir != 0) & (y_dir != 0));
        const float32_t scale = 1.0f - diag_mask * ONE_MINUS_INVSQRT2;
        player->velocity.x = x_dir * PLAYER_RUNSPEED * scale;
        player->velocity.y = y_dir * PLAYER_RUNSPEED * scale;

        /* Update position based on velocity and delta time */
        _apply_velocity(player, dt);
}