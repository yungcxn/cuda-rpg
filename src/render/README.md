# Note on rendering

- `render/tex.cu` holds `tex_tileline_t* (uint64_t*) devtilemap` and `tex_realrgba_t* (uint32_t*) devpalette`
- `devtilemap` is `res/res.h` array in a special formatted, laid out array of `tex_tileline_t` => holds palette indices
- `devpalette`is the palette, holds `tex_realrgba_t` at palette indices < 16

- `render/tileinfo.cu` holds `tileinfo_t* tileinfo_devtable` and `float32_t* tileinfo_animlen_devtable`
-  `tileinfo_devtable` holds tile indices to `tex-devtilemap` per tile id (16x16, for non-entities -> world)
-  `tileinfo_animlen_devtable` holds frame durations per tile id


- `render/spriteinfo.cu` holds `spriteinfo_t* spriteinfo_devtable` and `float32_t* spriteinfo_animlen_devtable`
- `spriteinfo_devtable` holds tile indices to `tex-devtilemap` per sprite id (16nx16n, for entities -> ecs)
- `spriteinfo_animlen_devtable` holds frame durations per sprite id