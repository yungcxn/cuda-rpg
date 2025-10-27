#ifndef WORLDHOME_H
#define WORLDHOME_H

#include "worldhome.h"
#include "../world.h"

world_t* worldhome_load(ecs_handle_t* ecs_handle);
void worldhome_update(world_ctx_t* world_ctx, float32_t dt);

#endif