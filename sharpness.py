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
    b, g, r = cv2.split(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_sobel_r = np.abs(cv2.Laplacian(r, cv2.CV_64F, ksize=1))
    img_sobel_g = np.abs(cv2.Laplacian(g, cv2.CV_64F, ksize=1))
    img_sobel_b = np.abs(cv2.Laplacian(b, cv2.CV_64F, ksize=1))
    img_sobel_gray = np.abs(cv2.Laplacian(img_gray, cv2.CV_64F, ksize=1))
    return (cv2.mean(img_sobel_r)[0], cv2.mean(img_sobel_g)[0], cv2.mean(img_sobel_b)[0], cv2.mean(img_sobel_gray)[0])


def proc(name, tile):
    letter = ["r", "g", "b", "gr"]
    image = cv2.imread(name)
    if image is None:
        return
    sharplist = " ".join([f"{x[0]}: {x[1]}" for x in list(zip(letter, laplacian(image)))])
    print(f"Average Sharpness for {name}  {sharplist}")
    height, width, _ = image.shape
    tmp_r = np.zeros(((height // tile) + 1, (width // tile) + 1))
    tmp_g = np.zeros(((height // tile) + 1, (width // tile) + 1))
    tmp_b = np.zeros(((height // tile) + 1, (width // tile) + 1))
    tmp_gray = np.zeros(((height // tile) + 1, (width // tile) + 1))

    for y in range(0, height, tile):
        for x in range(0, width, tile):
            crop_img = image[y : y + tile, x : x + tile]
            r, g, b, gray = laplacian(crop_img)
            tmp_r[y // tile][x // tile] = r
            tmp_g[y // tile][x // tile] = g
            tmp_b[y // tile][x // tile] = b
            tmp_gray[y // tile][x // tile] = gray

    for l, tmp in zip(letter, (tmp_r, tmp_g, tmp_b, tmp_gray)):
        plt.imshow(tmp, cmap="hot", interpolation="nearest")
        plt.colorbar()
        namew = Path(name).with_suffix("")
        print(f"Writing to {namew}-{l}-heatmap.png")
        plt.savefig(f"{namew}-{l}-heatmap.png")
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
