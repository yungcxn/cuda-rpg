

#include "ecs.h"

static ecs_t* ecs_instance;

void world_ctx_setup() {
    ecs_setup();
    ecs_instance = ecs_get();
}

void world_ctx_cleanup() {
    ecs_cleanup();
}

void world_ctx_update(float dt) {
    ecs_update(dt);
}
