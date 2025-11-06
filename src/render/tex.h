#ifndef TILE_H
#define TILE_H

#include <stdint.h>
#include "../res/res.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TEX_COLOURS 16
#define TEX_BITS_PER_COLOR 4 /* log_2 16 */
#define TEX_PIXELS_PER_64BIT_CHUNK 16
#define TEX_TILEWIDTH 16
#define TEX_TILEHEIGHT 16
#define TEX_64BIT_CHUNKS_PER_TILE TEX_PIXELS_PER_64BIT_CHUNK

#define TEX_TILEMAP_WIDTH RES_MOTHERSHEET_WIDTH_TILES
#define TEX_TILEMAP_HEIGHT RES_MOTHERSHEET_HEIGHT_TILES

typedef uint8_t tex_palref_t; /* 0-15 */
typedef uint64_t tex_tileline_t;
typedef uint32_t tex_realrgba_t; /* 0xRRGGBBAA */

#define TEX_TILELINE_INDEX(tile_x, tile_y, liny) \
        (TEX_TILEHEIGHT * TEX_TILEMAP_WIDTH * tile_y + TEX_TILEHEIGHT * tile_x + liny)

#define TEX_GET_PALREF_FROM_TEXLINE(texline, left_to_right_num) \
        ((tex_palref_t)(texline >> (((15 - (left_to_right_num)) << 2))) & 0xFULL)

tex_tileline_t* tex_devtilemap_create();
tex_realrgba_t* tex_devpalette_create();

void tex_devtilemap_destroy(tex_tileline_t* devtilemap);
void tex_devpalette_destroy(tex_realrgba_t* devpalette);

#ifdef __cplusplus
}
#endif

#endif
