#ifndef PLAYER_H
#define PLAYER_H

#include <stdint.h>
#include "../../headeronly/vec.h"
#include "../../render/spriteinfo.h"
#include "../../key.h"
#include "../bitfsm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
        spriteinfo_id_t spriteinfo;
        float32_t spritetimer;
} player_shared_t;

typedef struct {
        bitfsm_state_t state;
        vec2f32_t pos1;
        vec2f32_t pos2;
        vec2f32_t velocity;
        float32_t rolltimer;
        player_shared_t* shared;
} player_t;

player_t* player_create(void);
void player_destroy(player_t* player);
void player_setpos(player_t* player, vec2f32_t pos1);
void player_update(player_t* player, float32_t dt);
player_shared_t* player_shared_devbuf_create(void);
void player_shared_devbuf_destroy(player_shared_t* devbuf);
void player_bitfsm_callback(void* playerstate, bitfsm_token_t tok);

#ifdef __cplusplus
}
#endif

#endif
