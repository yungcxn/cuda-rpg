#ifndef PLAYER_H
#define PLAYER_H

#include <stdint.h>
#include "../../headeronly/vec.h"
#include "../../render/spriteinfo.h"
#include "../../key.h"

typedef struct {
        uint64_t state_hi;
        uint64_t state_lo;
        vec2f32_t pos1;
        vec2f32_t pos2;
        vec2f32_t velocity;
        spriteinfo_id_t spriteinfo;
} player_t;

player_t* player_init(void);
void player_destroy(player_t* player);
void player_setpos(player_t* player, vec2f32_t pos1);
void player_update(player_t* player, float32_t dt, key_inputfield_t pressed_keys);

#endif