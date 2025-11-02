#ifndef PLAYER_H
#define PLAYER_H

#include <stdint.h>
#include "../../headeronly/vec.h"
#include "../../render/spriteinfo.h"
#include "../../key.h"

typedef struct {
        spriteinfo_id_t spriteinfo;
        float32_t spritetimer;
} player_shared_t;

typedef struct {
        uint64_t state_hi;
        uint64_t state_lo;
        vec2f32_t pos1;
        vec2f32_t pos2;
        vec2f32_t velocity;
        player_shared_t* shared;
} player_t;

player_t* player_create(void);
void player_destroy(player_t* player);
void player_setpos(player_t* player, vec2f32_t pos1);
void player_update(player_t* player, float32_t dt, key_inputfield_t pressed_keys);
player_shared_t* player_shared_devbuf_create(void);
void player_shared_devbuf_destroy(player_shared_t* devbuf);

#endif
