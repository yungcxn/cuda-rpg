from pathlib import Path
import os
from PIL import Image
import numpy as np

res_h_path = Path(__file__).parent / "res.h"
PPT = 16
images = ["mothersheet.png"]
palette = "palette.png"

oldpath = Path.cwd()
os.chdir(Path(__file__).parent)

def generate_palette_dict(palette_image):
    img = Image.open(palette_image).convert("RGBA")
    palette_dict = {(0, 0, 0, 0): 0}   # transparent = index 0
    pixels = np.array(img)
    for i in range(1, img.width):
        color = tuple(pixels[0, i])
        palette_dict[color] = i
    return palette_dict

palette_dict = generate_palette_dict(palette)
print(palette_dict)

res_h_lines = ["#ifndef RES_H", "#define RES_H", ""]

# add to res_h_lines the palette in comma separated hex format to add into a C array
res_h_lines.append("#define RES_PALETTE_DATA \\")
# just add the keys of palette_dict in order of their index
palette_items = sorted(palette_dict.items(), key=lambda x: x[1])
for color, index in palette_items:
    res_h_lines.append(f"        0x{color[0]:02X}{color[1]:02X}{color[2]:02X}{color[3]:02X}, \\")
res_h_lines.append("\n")

for image_path in images:
    img = Image.open(image_path).convert("RGBA")
    pixels = np.array(img)
    if img.width % PPT != 0 or img.height % PPT != 0:
        raise ValueError(f"{image_path} dimensions must be divisible by {PPT}")
    
    width_tiles = img.width // PPT
    height_tiles = img.height // PPT
    var_name = Path(image_path).stem.upper() + "_DATA"
    
    # add macros for tile count
    res_h_lines.append(f"#define RES_{Path(image_path).stem.upper()}_WIDTH_TILES {width_tiles}")
    res_h_lines.append(f"#define RES_{Path(image_path).stem.upper()}_HEIGHT_TILES {height_tiles}\n")
    
    res_h_lines.append(f"#define RES_{Path(image_path).stem.upper()}_DATA \\")

    for ty in range(height_tiles):
        for tx in range(width_tiles):
            for py in range(PPT):
                tile_row = pixels[ty*PPT + py, tx*PPT:tx*PPT + PPT]
                value = 0
                for px, pixel in enumerate(tile_row):
                    idx = palette_dict.get(tuple(pixel), 0)
                    value |= (idx & 0xF) << (px*4)
                res_h_lines.append(f"        0x{value:016X}, \\")  # 8 spaces indentation
            res_h_lines.append("\\")  # line break after each tile
        res_h_lines.pop()  # remove last backslash
    res_h_lines.append("\n")

res_h_lines.append("#endif")

res_h_path.write_text("\n".join(res_h_lines))
os.chdir(oldpath)
print(f"Generated {res_h_path}")
