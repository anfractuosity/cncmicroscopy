#!/usr/bin/python3

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

"""

Simple script to use Fiji for image stitching
from our tiles.

Use integer values for x,y for tile coordinates for Fiji.
Flip image from camera and invert y coordinate.

"""


def invert_y(path, ext, vig=None):
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
        flip_remove_vignette(f, out, vig)

    return len(set(xs)), len(set(ys))

"""
Using code from:
https://stackoverflow.com/questions/74786867/subtract-vignetting-template-from-image-in-opencv-python
"""
def flip_remove_vignette(inf, outf, vigf=None):
    img1 = cv2.imread(inf)

    # Flip image vertically
    img1 = cv2.flip(img1, 0)

    # Remove vignette
    if vigf is not None:
        vig = cv2.imread(vigf, cv2.IMREAD_GRAYSCALE)  # Read vignette template as grayscale
        vig = cv2.medianBlur(vig, 15)  # Apply median filter for removing artifacts and extreem pixels.
        vig_norm = vig.astype(np.float32) / 255  # Convert vig to float32 in range [0, 1]
        vig_norm = cv2.GaussianBlur(vig_norm, (51, 51), 30)  # Blur the vignette template (because there are still artifacts, maybe because SO convered the image to JPEG).
        #vig_max_val = vig_norm.max()  # For avoiding "false colors" we may use the maximum instead of the mean.
        vig_mean_val = cv2.mean(vig_norm)[0]
        # vig_max_val / vig_norm
        inv_vig_norm = vig_mean_val / vig_norm  # Compute G = m/F
        inv_vig_norm = cv2.cvtColor(inv_vig_norm, cv2.COLOR_GRAY2BGR)  # Convert inv_vig_norm to 3 channels before using cv2.multiply. https://stackoverflow.com/a/48338932/4926757
        img2 = cv2.multiply(img1, inv_vig_norm, dtype=cv2.CV_8U)  # Compute: C = R * G
        cv2.imwrite(outf, img2)
    else:
        cv2.imwrite(outf, img1)

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="CLI to stitch the tiles with ImageJ")
    parser.add_argument("imagedir")
    parser.add_argument(
        "--ext", dest="ext", type=str, help="Image filename extension", default="jpg"
    )
    parser.add_argument('--vig', type=argparse.FileType('rb'), help="Vignette image", required=False)
    args = parser.parse_args()

    if not os.path.isdir(args.imagedir):
        print("Need to pass valid directory", file=sys.stderr)
        exit(1)

    vig = None
    if args.vig is not None:
        vig = args.vig.name

    gridx, gridy = invert_y(args.imagedir, args.ext, vig)

    script = f"""
	run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] grid_size_x={gridx} grid_size_y={gridy} tile_overlap=50 first_file_index_x=0 first_file_index_y=0 directory={args.imagedir} file_names=tileb_{{x}}_{{y}}.{args.ext} output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap computation_parameters=[Save memory (but be slower)] image_output=[Write to disk] output_directory={args.imagedir}");
	"""

    script_path = os.path.join(args.imagedir, "run.ijm")
    out = open(script_path, "w")
    out.write(script)
    out.close()

    os.system(f"ImageJ-linux64 --headless --console -macro {script_path}")
