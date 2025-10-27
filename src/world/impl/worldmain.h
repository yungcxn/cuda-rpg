#ifndef WORLDMAIN_H
#define WORLDMAIN_H

#include "worldmain.h"
#include "../world.h"

world_t* worldmain_load(ecs_handle_t* ecs_handle);
void worldmain_update(world_ctx_t* world_ctx, float32_t dt);

#endif