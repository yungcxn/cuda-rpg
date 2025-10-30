#ifndef TILE_H
#define TILE_H

#include <stdint.h>
#include "../headeronly/vec.h"

#define TEX_COLOURS 16
#define TEX_BITS_PER_COLOR 4 /* log_2 16 */
#define TEX_PIXELS_PER_64BIT_CHUNK (64 / TEX_BITS_PER_COLOR) /* 16 */
#define TEX_64BIT_CHUNKS_PER_TILE TEX_PIXELS_PER_64BIT_CHUNK

#define TEX_TILEMAP_GET_LINE0_IDX(tileidx, tileidy) \
        (((tileidy) * TEX_64BIT_CHUNKS_PER_TILE) + (tileidx))

typedef uint64_t tex_tileline_t;
typedef uint32_t tex_realrgba_t; /* 0xRRGGBBAA */

void tex_devdata_init();
void tex_devdata_cleanup();

#endif