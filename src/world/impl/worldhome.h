#ifndef WORLDHOME_H
#define WORLDHOME_H

#include "../world.h"

#ifdef __cplusplus
extern "C" {
#endif

void worldhome_load(world_ctx_t* world_ctx);
void worldhome_update(world_ctx_t* world_ctx, float64_t dt);

#ifdef __cplusplus
}
#endif

#endif