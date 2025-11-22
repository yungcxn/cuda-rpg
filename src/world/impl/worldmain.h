#ifndef WORLDMAIN_H
#define WORLDMAIN_H

#include "../world.h"

#ifdef __cplusplus
extern "C" {
#endif

void worldmain_load(world_ctx_t* world_ctx);
void worldmain_update(world_ctx_t* world_ctx, float64_t dt);

#ifdef __cplusplus
}
#endif

#endif