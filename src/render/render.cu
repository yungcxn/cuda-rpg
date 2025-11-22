
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sys/types.h>
#include <unistd.h>
#include <cfloat>

#include "render.h"
#include "vulkan.h"
#include "../headeronly/cuarray.cuh"
#include "../headeronly/def.cuh"

#include "../headeronly/def.h"
#include "../headeronly/gamemeta.h"
#include "spriteinfo.h"
#include "tileinfo.h"
#include "tex.h"
#include "../world/ecs.h"
#include "../world/world.h"
#include "../world/player/player.h"

#define RENDER_DEFAULTBLOCKSIZE 256
#define RENDER_OFFSCREEN_SPRITEMARGIN 10 /* tiles */
#define RENDER_TRANSLATED_PLAYER_ENTITYID (ECS_MAX_ENTITIES)
#define RENDER_MINDEPTH FLT_MIN
#define RENDER_MAXDEPTH FLT_MAX

static tex_tileline_t* tex_devtilemap;
static tex_realrgba_t* tex_devpalette;
static tileinfo_t* tileinfo_devtable;
static float32_t* tileinfo_animlen_devtable;
static float32_t* tileinfo_animtimer_devcache;
static uint32_t* tileinfo_animtimer_updatelocks;
static spriteinfo_t* spriteinfo_devtable;
static float32_t* spriteinfo_animlen_devtable;
static spriteinfo_bboff_t* spriteinfo_bboff_devtable;

static player_shared_t* devplayer;
static ecs_shared_t* devecs;
static vec2f32_t* devecs_pos1;
static cuarray_t<uint32_t>* devrenderable_ids;

static cudaTextureObject_t texObj_tex_tilemap;                   /* tex_tileline_t* */
static cudaTextureObject_t texObj_tex_palette;                   /* tex_realrgba_t* */
static cudaTextureObject_t texObj_tileinfo_devtable;             /* tileinfo_t* */
static cudaTextureObject_t texObj_tileinfo_animlen_devtable;     /* float32_t* */
static cudaTextureObject_t texObj_spriteinfo_devtable;           /* spriteinfo_t* */
static cudaTextureObject_t texObj_spriteinfo_animlen_devtable;   /* float32_t* */
static cudaTextureObject_t texObj_spriteinfo_bboff_devtable;     /* spriteinfo_bboff_t* */

static float32_t* depthbuffer;

static vec2f32_t render_hostcamerapos = {0.0f, 0.0f};

void render_data_setup(void) {

        /* here we create all dev data to access */
        /* aswell as buffers for devreleases */
        tex_devtilemap = tex_devtilemap_create();
        tex_devpalette = tex_devpalette_create();

        tileinfo_devtable = tileinfo_devtable_create();
        tileinfo_animlen_devtable = tileinfo_animlen_devtable_create();
        tileinfo_animtimer_devcache = tileinfo_animtimer_devcache_create();
        cudaMalloc(&tileinfo_animtimer_updatelocks, sizeof(uint32_t) * TILEINFOS);

        spriteinfo_devtable = spriteinfo_devtable_create();
        spriteinfo_animlen_devtable = spriteinfo_animlen_devtable_create();
        spriteinfo_bboff_devtable = spriteinfo_bboff_devtable_create();

        cudaResourceDesc resDesc = {};
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = tex_devtilemap;
        resDesc.res.linear.sizeInBytes = sizeof(tex_tileline_t) * TEX_TILEMAP_WIDTH 
                                                                * TEX_TILEMAP_HEIGHT 
                                                                * TEX_64BIT_CHUNKS_PER_TILE;
        resDesc.res.linear.desc = cuda_create_channel_desc_custom<tex_tileline_t>();
        cudaCreateTextureObject(&texObj_tex_tilemap, &resDesc, &texDesc, nullptr);

        resDesc.res.linear.devPtr = tex_devpalette;
        resDesc.res.linear.sizeInBytes = sizeof(tex_realrgba_t) * TEX_COLOURS;
        resDesc.res.linear.desc = cudaCreateChannelDesc<tex_realrgba_t>();
        cudaCreateTextureObject(&texObj_tex_palette, &resDesc, &texDesc, nullptr);

        resDesc.res.linear.devPtr = tileinfo_devtable;
        resDesc.res.linear.sizeInBytes = sizeof(tileinfo_t) * TILEINFOS;
        resDesc.res.linear.desc = cuda_create_channel_desc_custom<tileinfo_t>(); /* 32 bit */
        cudaCreateTextureObject(&texObj_tileinfo_devtable, &resDesc, &texDesc, nullptr);

        resDesc.res.linear.devPtr = tileinfo_animlen_devtable;
        resDesc.res.linear.sizeInBytes = sizeof(float32_t) * TILEINFOS;
        resDesc.res.linear.desc = cudaCreateChannelDesc<float32_t>();
        cudaCreateTextureObject(&texObj_tileinfo_animlen_devtable, &resDesc, &texDesc, nullptr);

        resDesc.res.linear.devPtr = spriteinfo_devtable;
        resDesc.res.linear.sizeInBytes = sizeof(spriteinfo_t) * SPRITEINFOS;
        resDesc.res.linear.desc = cuda_create_channel_desc_custom<spriteinfo_t>();
        cudaCreateTextureObject(&texObj_spriteinfo_devtable, &resDesc, &texDesc, nullptr);

        resDesc.res.linear.devPtr = spriteinfo_animlen_devtable;
        resDesc.res.linear.sizeInBytes = sizeof(float32_t) * SPRITEINFOS;
        resDesc.res.linear.desc = cudaCreateChannelDesc<float32_t>();
        cudaCreateTextureObject(&texObj_spriteinfo_animlen_devtable, &resDesc, &texDesc, nullptr);

        resDesc.res.linear.devPtr = spriteinfo_bboff_devtable;
        resDesc.res.linear.sizeInBytes = sizeof(spriteinfo_bboff_t) * SPRITEINFOS;
        resDesc.res.linear.desc = cuda_create_channel_desc_custom<spriteinfo_bboff_t>();
        cudaCreateTextureObject(&texObj_spriteinfo_bboff_devtable, &resDesc, &texDesc, nullptr);

        devplayer = player_shared_devbuf_create();
        devecs = ecs_shared_devbuf_create();
        devecs_pos1 = ecs_pos1_devbuf_create();
        devrenderable_ids = cuarray_create<uint32_t>(ECS_MAX_ENTITIES + 1); /* +1 for player */

        cudaMalloc(&depthbuffer, sizeof(float32_t) * WIDTH * HEIGHT);
}

void render_data_cleanup(void) {
        /* all above mentioned buffers are released here */
        cudaDestroyTextureObject(texObj_tex_tilemap);
        cudaDestroyTextureObject(texObj_tex_palette);
        cudaDestroyTextureObject(texObj_tileinfo_devtable);
        cudaDestroyTextureObject(texObj_tileinfo_animlen_devtable);
        cudaDestroyTextureObject(texObj_spriteinfo_devtable);
        cudaDestroyTextureObject(texObj_spriteinfo_animlen_devtable);
        cudaDestroyTextureObject(texObj_spriteinfo_bboff_devtable);

        tex_devpalette_destroy(tex_devpalette);
        tileinfo_devtable_destroy(tileinfo_devtable);
        tileinfo_animlen_devtable_destroy(tileinfo_animlen_devtable);

        tileinfo_animtimer_devcache_destroy(tileinfo_animtimer_devcache);
        cudaFree(tileinfo_animtimer_updatelocks);

        spriteinfo_devtable_destroy(spriteinfo_devtable);
        spriteinfo_animlen_devtable_destroy(spriteinfo_animlen_devtable);
        spriteinfo_bboff_devtable_destroy(spriteinfo_bboff_devtable);
        player_shared_devbuf_destroy(devplayer);
        ecs_shared_devbuf_destroy(devecs);
        ecs_pos1_devbuf_destroy(devecs_pos1);
        cuarray_destroy(devrenderable_ids);

        cudaFree(depthbuffer);
}

__global__ 
void _k_collect_renderable_sprites(ecs_shared_t* devecs, vec2f32_t* devecs_pos1,
                                        player_shared_t* devplayer, vec2f32_t player_pos,
                                        vec2f32_t cam, cuarray_t<uint32_t>* renderable_ids) {

        /* id corresponds to entityid */
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        vec2f32_t pos;
        spriteinfo_id_t id;
        if (idx == RENDER_TRANSLATED_PLAYER_ENTITYID) { /* player thread */
                pos = player_pos;
                id = devplayer->spriteinfo;
        } else if (idx < ECS_MAX_ENTITIES) {
                pos = devecs_pos1[idx];
                id = devecs->spriteinfos[idx];
        } else {
                return;
        }

        if ((pos.x == -1.0f && pos.y == -1.0f) 
            || pos.x < (cam.x - RENDER_OFFSCREEN_SPRITEMARGIN) 
            || pos.x > (cam.x + RENDER_OFFSCREEN_SPRITEMARGIN + WIDTH_TILES) 
            || pos.y < (cam.y - RENDER_OFFSCREEN_SPRITEMARGIN) 
            || pos.y > (cam.y + RENDER_OFFSCREEN_SPRITEMARGIN + HEIGHT_TILES)
            || id == SID(NONE)) {
                return; /* out of render bounds */
        }

        cuarray_concadd(renderable_ids, idx);
}

__global__ 
void _k_animate_renderable_sprites(ecs_shared_t* devecs, player_shared_t* devplayer, 
                                   cuarray_t<uint32_t>* renderable_ids,
                                   float32_t* spriteinfo_animlen_devtable, const float64_t dt){

        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= cuarray_length(renderable_ids)) return;

        /* update sprite animation frame based on dt */
        const uint32_t entity_id = cuarray_get(renderable_ids, idx);
        spriteinfo_id_t id;
        float32_t oldtime;
        spriteinfo_id_t* spriteinfo_ptr;
        float32_t* spritetimer_ptr;
        if (entity_id == RENDER_TRANSLATED_PLAYER_ENTITYID) {
                oldtime = devplayer->spritetimer;
                if (oldtime == SPRITEINFO_ANIM_NOANIMDURATION) return; /* unanimated */
                /* player animation */
                id = devplayer->spriteinfo;
                spriteinfo_ptr = &devplayer->spriteinfo;
                spritetimer_ptr = &devplayer->spritetimer;
        } else if (entity_id < ECS_MAX_ENTITIES) {
                oldtime = devecs->spritetimers[entity_id];
                if (oldtime == SPRITEINFO_ANIM_NOANIMDURATION) return; /* unanimated */
                /* ecs animation */
                id = devecs->spriteinfos[entity_id];
                spriteinfo_ptr = &devecs->spriteinfos[entity_id];
                spritetimer_ptr = &devecs->spritetimers[entity_id];
        }
        const float32_t animlen = spriteinfo_animlen_devtable[id];
        *spritetimer_ptr += dt;
        if (*spritetimer_ptr >= animlen) {
                *spritetimer_ptr = 0.0f;
                *spriteinfo_ptr = (id + 1);
                const float32_t newanimlen = spriteinfo_animlen_devtable[id + 1];
                if (newanimlen < 0.0f) { /* sentinel reached */
                        *spriteinfo_ptr += newanimlen; /* loop back */
                }
        }
}

__global__ 
static void _k_animate_worldmap_slice(world_map_devdata_t mapdata, 
                                      cudaTextureObject_t texObj_tileinfo_animlen_devtable,
                                      float32_t* tileinfo_animtimer_devcache, 
                                      uint32_t* tileinfo_animtimer_updatelocks, 
                                      const vec2i32_t coarse_cam, const float64_t dt) {

        /* one thread per tile per layer */
        /* started 1d setup */
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        tileinfo_id_t* layer = 0;
        {
                const uint32_t layer_id = (idx / (PADDEDWIDTH_TILES * PADDEDHEIGHT_TILES));
                switch (layer_id) {
                        case 0: layer = mapdata.bg; break;
                        case 1: layer = mapdata.main; break;
                        case 2: layer = mapdata.on_main; break;
                        case 3: layer = mapdata.fg; break;
                        default: return;
                }
        }

        if (!layer) return; /* layer not present */

        uint32_t layer_tile_id = 0;
        {
                const uint32_t world_tile_idx = (idx % (PADDEDWIDTH_TILES)) + coarse_cam.x;
                const uint32_t world_tile_idy = (idx / (PADDEDWIDTH_TILES)) 
                                                % (PADDEDHEIGHT_TILES) + coarse_cam.y;

                if (world_tile_idy >= mapdata.dim.height || world_tile_idx >= mapdata.dim.width) 
                        return;

                layer_tile_id = world_tile_idy * mapdata.dim.width + world_tile_idx;
        }
        
        tileinfo_id_t id = layer[layer_tile_id];
        const float32_t animlen = tex1Dfetch<float32_t>(texObj_tileinfo_animlen_devtable, id);
        if (animlen == SPRITEINFO_ANIM_NOANIMDURATION) return; /* unanimated */

        {
                /* achieve lock atomically, will be reset next run by host */
                uint32_t old = atomicCAS(&tileinfo_animtimer_updatelocks[id], 0, 1);
                if (old == 0) { /* we got the lock */
                        tileinfo_animtimer_devcache[id] += dt;
                        __threadfence();
                        /* 2 here means: update released */
                        atomicExch(&tileinfo_animtimer_updatelocks[id], 2);
                } else {
                        while (atomicAdd(&tileinfo_animtimer_updatelocks[id], 0) != 2) {
                                /* wait until update is released */
                        }
                }
                /* this solves the issue of visibility, meaning all threads proceed after update */
                /*   and only one is doing it through atomic locks                               */
        }

        if (tileinfo_animtimer_devcache[id] >= animlen) {
                tileinfo_animtimer_devcache[id] = 0.0f;
                id++;
                const float32_t newanimlen = tex1Dfetch<float32_t>(texObj_tileinfo_animlen_devtable,   
                                                                   id);

                if (newanimlen < 0.0f) id += newanimlen; /* loop back for reached sentinel */
                layer[layer_tile_id] = id;
        }

}

__global__ 
void _k_clear_depth(float32_t* screen_depthbuffer) {
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < WIDTH && y < HEIGHT) screen_depthbuffer[y * WIDTH + x] = RENDER_MINDEPTH;
}

static inline void _prerender(world_map_devdata_t devworldmap, ecs_handle_t hostecs_handle,
                              player_t* hostplayer, const float64_t dt) {
                                
        cudaMemcpy((void*)devecs, (const void*)hostecs_handle.instance->shared, 
                   sizeof(ecs_shared_t), cudaMemcpyHostToDevice);

        cudaMemcpy((void*)devplayer, (const void*)hostplayer->shared, sizeof(player_shared_t),
                   cudaMemcpyHostToDevice);

        cudaMemcpy((void*)devecs_pos1, (const void*)hostecs_handle.instance->pos1, 
                   sizeof(vec2f32_t) * ECS_MAX_ENTITIES, cudaMemcpyHostToDevice);

        cuarray_hostclear(devrenderable_ids);

        {
                dim3 blockDim(16, 16);
                dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, 
                             (HEIGHT + blockDim.y - 1) / blockDim.y);

                _k_clear_depth<<<gridDim, blockDim>>>(depthbuffer);
        }

        {
                cudaMemset(tileinfo_animtimer_updatelocks, 0, sizeof(uint32_t) * TILEINFOS);

                constexpr uint32_t threadsperblock = PADDEDWIDTH_TILES * PADDEDHEIGHT_TILES;
                constexpr uint32_t numblocks = WORLD_MAP_LAYERS;

                vec2i32_t coarse_cam = {
                        (int32_t)(render_hostcamerapos.x),
                        (int32_t)(render_hostcamerapos.y)
                };
                _k_animate_worldmap_slice<<<numblocks, threadsperblock>>>(devworldmap, 
                                                                   texObj_tileinfo_animlen_devtable,
                                                                        tileinfo_animtimer_devcache,
                                                                     tileinfo_animtimer_updatelocks,
                                                                          coarse_cam, dt);
        }

        {
                /* +1 for player but emitted since we cannot use higher than 1024 threads */
                constexpr uint32_t threads = RENDER_DEFAULTBLOCKSIZE;
                const uint32_t total_threads = ECS_MAX_ENTITIES + 1;
                const uint32_t numblocks = (total_threads + threads - 1) / threads;
                _k_collect_renderable_sprites<<<numblocks, threads>>>(devecs, devecs_pos1, 
                                                                      devplayer, 
                                                                      hostplayer->pos1,
                                                                      render_hostcamerapos,
                                                                      devrenderable_ids);
        }

        {
                uint32_t renderablecount = cuarray_hostlength(devrenderable_ids);
                const uint32_t total_threads = renderablecount;
                constexpr uint32_t blocksize = RENDER_DEFAULTBLOCKSIZE;
                const uint32_t numblocks = (total_threads + blocksize - 1) / blocksize;
                _k_animate_renderable_sprites<<<numblocks, blocksize>>>(devecs, devplayer,
                                                                        devrenderable_ids,
                                                                        spriteinfo_animlen_devtable,
                                                                        dt);
        }
}

static inline void _postrender(ecs_shared_t* hostecs, player_shared_t* hostplayer) {
        cudaMemcpy((void*)hostecs, (const void*)devecs, sizeof(ecs_shared_t), 
                   cudaMemcpyDeviceToHost);

        cudaMemcpy((void*)hostplayer, (const void*)devplayer, sizeof(player_shared_t),
                   cudaMemcpyDeviceToHost);
}

__device__ __forceinline__ 
static tex_palref_t _get_tile_palref(cudaTextureObject_t texObj_tileinfo_devtable, 
                                     cudaTextureObject_t texObj_tex_tilemap, tileinfo_id_t tileid, 
                                     uint32_t px, uint32_t py) {

        if (tileid == TID(VOID)) return 0;

        const tileinfo_t tileinfo = tex_1d_fetch_custom<tileinfo_t>(texObj_tileinfo_devtable,
                                                                    tileid);

        const uint32_t tilelineindex = TEX_TILELINE_INDEX(tileinfo.tx, tileinfo.ty, py);

        const tex_tileline_t line = tex_1d_fetch_custom<tex_tileline_t>(texObj_tex_tilemap,
                                                                        tilelineindex);
        return TEX_GET_PALREF_FROM_TEXLINE(line, px);
}

__device__ __forceinline__
void _render_pixel(uint32_t* framebuffer, float32_t* screen_depthbuffer, 
                   const tex_realrgba_t* palette, vec2i32_t screen_pos, 
                   float32_t depth, tex_palref_t palref) {
        uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
        int index = screen_pos.y * WIDTH + screen_pos.x;
        float32_t old_depth = atomicAdd(&screen_depthbuffer[index], 0.0f);

        if (depth > old_depth) {
                uint32_t old_uint = __float_as_uint(old_depth);
                uint32_t depth_uint = __float_as_uint(depth);
                uint32_t prev = atomicCAS((uint32_t*)&screen_depthbuffer[index], old_uint,
                                          depth_uint);
                if (prev == old_uint) {
                        framebuffer[index] = palette[palref];
                }
        }
}


__global__ 
static void _k_render_world_map(tex_realrgba_t* framebuffer, world_map_devdata_t mapdata, 
                                cudaTextureObject_t texObj_tileinfo_devtable, 
                                cudaTextureObject_t texObj_tex_tilemap, 
                                cudaTextureObject_t texObj_tex_palette, 
                                float32_t* depthbuffer, vec2i32_t coarse_cam,
                                vec2i32_t pixeloffset) {

        constexpr float32_t depths_per_layer[] = {
                RENDER_MINDEPTH + 1.0f, /* bg */
                RENDER_MINDEPTH + 2.0f, /* main */
                RENDER_MAXDEPTH - 1.0f, /* on_main */
                RENDER_MAXDEPTH, /* fg */
        };

        __shared__ tex_realrgba_t shared_palette[TEX_COLOURS];

        /* save palette */
        {
                const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
                if (tid < TEX_COLOURS) {
                        shared_palette[tid] = 
                                tex_1d_fetch_custom<tex_realrgba_t>(texObj_tex_palette, tid);
                }
        }
        
        __syncthreads();

        /* world tile coords, which have its width / 4 */
        /* cam 1.0f is one tile */
        const int32_t world_tile_x = coarse_cam.x + blockIdx.x;
        const int32_t world_tile_y = coarse_cam.y + blockIdx.y;

        /* we do not want to write below or above world boundaries */
        if (world_tile_y < 0 || world_tile_y >= mapdata.dim.height
                || world_tile_x < 0 || world_tile_x >= mapdata.dim.width) {
                return;
        }

        tex_palref_t final_pal = 0;
        float32_t final_depth = 0;
        if (mapdata.fg != 0) {
                final_pal = _get_tile_palref(texObj_tileinfo_devtable, texObj_tex_tilemap, 
                                       mapdata.fg[world_tile_y * mapdata.dim.width + world_tile_x],
                                             threadIdx.x, threadIdx.y);

                if (final_pal != 0) {
                        final_depth = depths_per_layer[3];
                        goto draw;
                }
        }

        if (mapdata.on_main != 0) {
                final_pal = _get_tile_palref(texObj_tileinfo_devtable, texObj_tex_tilemap, 
                                  mapdata.on_main[world_tile_y * mapdata.dim.width + world_tile_x],
                                             threadIdx.x, threadIdx.y);

                if (final_pal != 0) {
                        final_depth = depths_per_layer[2];
                        goto draw;
                }
        }

        if (mapdata.main != 0) {
                final_pal = _get_tile_palref(texObj_tileinfo_devtable, texObj_tex_tilemap, 
                                     mapdata.main[world_tile_y * mapdata.dim.width + world_tile_x],
                                             threadIdx.x, threadIdx.y);
                if (final_pal != 0) {
                        final_depth = depths_per_layer[1];
                        goto draw;
                }
        }

        if (mapdata.bg != 0) {
                final_pal = _get_tile_palref(texObj_tileinfo_devtable, texObj_tex_tilemap, 
                                       mapdata.bg[world_tile_y * mapdata.dim.width + world_tile_x],
                                             threadIdx.x, threadIdx.y);

                if (final_pal != 0) {
                        final_depth = depths_per_layer[0];
                        goto draw;
                }
        }

        return;
draw:
        const int32_t screen_x = (blockIdx.x * TEX_TILEWIDTH) + threadIdx.x - pixeloffset.x;
        const int32_t screen_y = (blockIdx.y * TEX_TILEHEIGHT) + threadIdx.y - pixeloffset.y;

        /* screen check */
        if (screen_x < 0 || screen_x >= WIDTH || screen_y < 0 || screen_y >= HEIGHT)
                return;

        _render_pixel(framebuffer, depthbuffer, shared_palette, 
                      {screen_x, screen_y}, 
                      final_depth, final_pal);
}

__global__ 
static void _k_render_entity_sprites(tex_realrgba_t* framebuffer, ecs_shared_t* ecs, 
                                          vec2f32_t* ecs_pos1, player_shared_t* player, 
                                          vec2f32_t player_pos1, 
                                          cudaTextureObject_t texObj_spriteinfo_devtable,
                                          cudaTextureObject_t texObj_tex_tilemap,
                                          cudaTextureObject_t texObj_tex_palette,
                                          cudaTextureObject_t texObj_spriteinfo_bboff_devtable,
                                          float32_t* depthbuffer, 
                                          cuarray_t<uint32_t>* devrenderable_ids, vec2f32_t cam) {

        __shared__ tex_realrgba_t shared_palette[TEX_COLOURS];

        {
                const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
                if (tid < TEX_COLOURS) {
                        shared_palette[tid] = 
                                tex_1d_fetch_custom<tex_realrgba_t>(texObj_tex_palette, tid);
                }
        }
        __syncthreads();

        /* TILEWIDTH x TILEHIEGHT threads per tile, so block is tile id */
        const uint32_t tile_id = blockIdx.x;
        if (tile_id >= cuarray_length(devrenderable_ids)) return;

        const uint32_t entity_id = cuarray_get(devrenderable_ids, tile_id);
        vec2f32_t pos;
        spriteinfo_id_t sprite_id;

        if (entity_id == RENDER_TRANSLATED_PLAYER_ENTITYID) { /* player */
                pos = player_pos1;
                sprite_id = player->spriteinfo;
        } else if (entity_id < ECS_MAX_ENTITIES) {
                pos = ecs_pos1[entity_id];
                sprite_id = ecs->spriteinfos[entity_id];
        } else {
                return;
        }

        const spriteinfo_t spriteinfo = tex_1d_fetch_custom<spriteinfo_t>(
                texObj_spriteinfo_devtable, sprite_id
        );
        const spriteinfo_bboff_t bboff = tex_1d_fetch_custom<spriteinfo_bboff_t>(
                texObj_spriteinfo_bboff_devtable, sprite_id
        );
        
        for (uint32_t ty = 0; ty < spriteinfo.th; ty++) {
                for (uint32_t tx = 0; tx < spriteinfo.tw; tx++) {
                        const int32_t screen_x = (int32_t)((pos.x - cam.x) * TEX_TILEWIDTH) 
                                                 + (int32_t)(tx * TEX_TILEWIDTH) 
                                                 + (int32_t)(bboff.x) 
                                                 + (int32_t)(threadIdx.x);
                        const int32_t screen_y = (int32_t)((pos.y - cam.y) * TEX_TILEHEIGHT) 
                                                 + (int32_t)(ty * TEX_TILEHEIGHT) 
                                                 + (int32_t)(bboff.y) 
                                                 + (int32_t)(threadIdx.y);

                        /* screen check */
                        if (screen_x < 0 || screen_x >= WIDTH 
                            || screen_y < 0 || screen_y >= HEIGHT)
                                continue;

                        const uint32_t tilelineindex = TEX_TILELINE_INDEX(spriteinfo.tx + tx,
                                                                          spriteinfo.ty + ty,
                                                                          threadIdx.y);

                        const tex_tileline_t line = 
                                tex_1d_fetch_custom<tex_tileline_t>(texObj_tex_tilemap,
                                                                    tilelineindex);

                        const tex_palref_t palref = TEX_GET_PALREF_FROM_TEXLINE(line, threadIdx.x);

                        if (palref != 0) {
                                _render_pixel(framebuffer, depthbuffer, shared_palette,
                                              {screen_x, screen_y}, pos.y + pos.x * 0.0001f,
                                              palref);
                        }
                }
        }
}

/* pixel write is safe; therefore we can render map and entities at the same time. */
void render(tex_realrgba_t** framebuffer_ptr, world_map_devdata_t devmapdata, 
            ecs_handle_t hostecs_handle, player_t* hostplayer, float64_t dt) {

        tex_realrgba_t* framebuffer = *framebuffer_ptr; /* after vulkan post, this gets updated */
        uint32_t img_idx = 0;
        vulkan_pre_render(&img_idx);

        _prerender(devmapdata, hostecs_handle, hostplayer, dt);
        cudaDeviceSynchronize();
        
        {
                const vec2f32_t cam_f = render_hostcamerapos;
                const vec2i32_t cam_i = {(int32_t)cam_f.x, (int32_t)cam_f.y};
                const vec2i32_t pixeloffset = {
                        (int32_t)((cam_f.x - (float32_t)cam_i.x) * TEX_TILEWIDTH),
                        (int32_t)((cam_f.y - (float32_t)cam_i.y) * TEX_TILEHEIGHT)
                };

                dim3 gridDim(PADDEDWIDTH_TILES, PADDEDHEIGHT_TILES, 1); /* tile amount of blocks */
                dim3 blockDim(TEX_TILEWIDTH, TEX_TILEHEIGHT, 1);

                _k_render_world_map<<<gridDim, blockDim>>>(framebuffer, devmapdata,
                                                                texObj_tileinfo_devtable,
                                                                texObj_tex_tilemap,
                                                                texObj_tex_palette,
                                                                depthbuffer,
                                                                cam_i, pixeloffset);
        }

        {
                uint32_t host_renderables = cuarray_hostlength(devrenderable_ids);

                dim3 blockDim(TEX_TILEWIDTH, TEX_TILEHEIGHT, 1);
                dim3 gridDim(host_renderables, 1, 1);
                _k_render_entity_sprites<<<gridDim, blockDim>>>(framebuffer, devecs, 
                                                                     devecs_pos1, devplayer,
                                                                     hostplayer->pos1,
                                                                     texObj_spriteinfo_devtable,
                                                                     texObj_tex_tilemap,
                                                                     texObj_tex_palette,
                                                                   texObj_spriteinfo_bboff_devtable,
                                                                     depthbuffer,
                                                                     devrenderable_ids,
                                                                     render_hostcamerapos);
        }

        cudaDeviceSynchronize();

        _postrender(hostecs_handle.instance->shared, hostplayer->shared);
        vulkan_post_render(img_idx);
}
