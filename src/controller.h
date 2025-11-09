#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "./world/world.h"
#include "./headeronly/def.h"
#include "key.h"

void controller_apply(world_ctx_t* world_ctx, key_inputfield_t pressed_keys, 
                      key_inputfield_t released_keys);

#endif