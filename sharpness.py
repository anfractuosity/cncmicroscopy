#!/usr/bin/python3

"""
    Generate heatmap showing sharpest tiles of image
"""
import argparse
import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Found Laplacian function here:
# https://github.com/ArduCAM/RaspberryPi/blob/master/Motorized_Focus_Camera/python/Autofocus.py#L14
def laplacian(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_sobel = np.abs(cv2.Laplacian(img_gray, cv2.CV_16S, ksize=3))
    return cv2.mean(img_sobel)[0]


def proc(name, tile):
    image = cv2.imread(name)
    if image is None:
        return
    height, width, _ = image.shape
    tmp = np.zeros(((height // tile) + 1, (width // tile) + 1))

    for y in range(0, height, tile):
        for x in range(0, width, tile):
            crop_img = image[y : y + tile, x : x + tile]
            tmp[y // tile][x // tile] = laplacian(crop_img)

    plt.imshow(tmp, cmap="hot", interpolation="nearest")
    plt.colorbar()
    namew = Path(name).with_suffix("")
    print(f"Writing to {namew}-heatmap.png")
    plt.savefig(f"{namew}-heatmap.png")
    plt.close()


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate heatmap showing sharpest tiles of image"
    )
    parser.add_argument(
        "--path", dest="path", type=str, help="Path to images", required=True
    )
    parser.add_argument(
        "--size", dest="size", type=int, help="Size of tile in pixels", required=True
    )
    args = parser.parse_args()
    for file in glob.glob(os.path.expanduser(args.path)):
        proc(file, args.size)
