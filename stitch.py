#!/usr/bin/python3

import glob
import os
from pathlib import Path

"""

Simple script to use Fiji for image stitching
from our tiles.

"""

image_dir = "/tmp/images"
ext = "tif"

"""

Use integer values for x,y for tile coordinates for Fiji.
Flip image from camera and invert y coordinate.

"""
def invert_y(path, ext):
    xs = []
    ys = []
    repx = {}
    repy = {}

    for f in glob.glob(os.path.join(path, f"tile_*.{ext}")):
        _, x, y = Path(f).stem.split("_")
        xs += [float(x)]
        ys += [float(y)]

    for idx, x in enumerate(sorted(set(xs))):
        repx[x] = idx

    y_ = len(sorted(set(ys))) - 1
    for y in sorted(set(ys)):
        repy[y] = y_
        y_ -= 1

    for f in glob.glob(os.path.join(path, f"tile_*.{ext}")):
        _, x, y = Path(f).stem.split("_")
        out = os.path.join(path, f"tileb_{repx[float(x)]}_{repy[float(y)]}.{ext}")
        os.system(f"magick {f} -flip {out}")

    return len(set(xs)), len(set(ys))

gridx, gridy = invert_y(image_dir, ext)

script = f"""
run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] grid_size_x={gridx} grid_size_y={gridy} tile_overlap=50 first_file_index_x=0 first_file_index_y=0 directory={image_dir} file_names=tileb_{{x}}_{{y}}.tif output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap computation_parameters=[Save memory (but be slower)] image_output=[Write to disk] output_directory={image_dir}");
"""

script_path = os.path.join(image_dir, "run.ijm")
out = open(script_path, "w")
out.write(script)
out.close()

os.system(f"ImageJ-linux64 --headless --console -macro {script_path}")