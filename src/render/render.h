#ifndef RENDER_CUH
#define RENDER_CUH

#include <X11/Xlib.h>
#include "../headeronly/def.h"
#include "../world/world.h"
#include "../world/ecs.h"
#include "../world/player/player.h"
#include "tex.h"

#ifdef __cplusplus
extern "C" {
#endif

void render_data_setup(void);
void render_data_cleanup(void);
void render(tex_realrgba_t** framebuffer_ptr, world_map_devdata_t devmapdata, 
            ecs_handle_t hostecs_handle, player_t* hostplayer, float32_t dt);

#ifdef __cplusplus
}
#endif

#endif