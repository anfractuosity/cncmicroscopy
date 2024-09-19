#!/usr/bin/python3

import math
import os
import queue
import re
import statistics
import time
from collections import deque
from concurrent.futures import Future
from io import BytesIO
from multiprocessing import Queue
from threading import Thread

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import serial
from PIL import Image

import toupcam


# Found Laplacian function originally here:
# https://github.com/ArduCAM/RaspberryPi/blob/master/Motorized_Focus_Camera/python/Autofocus.py#L14
#
# Note to get it to use the correct kernel for me, I had to switch from ksize=3 to ksize=1
# based on reading https://stackoverflow.com/questions/72612991/what-is-the-kernel-used-in-opencv-cv2-laplacian-function
#
# Reading https://www.projectpro.io/recipes/what-are-laplacian-derivatives-of-image-opencv
# I changed to cv2.CV_64F from CV_16S
def laplacian(img):
    img_sobel = np.abs(cv2.Laplacian(img, cv2.CV_64F, ksize=1))
    return cv2.mean(img_sobel)[0]


class Frame:
    def __init__(self, buf, width, height):
        self.buf = buf
        self.timestamp = time.time()
        self.width = width
        self.height = height

class CNCMicroscope:
    def __init__(
        self,
        begin,
        end,
        squaresize=0.5,
        zlimit=(-1, -11),
        zaf=(-1, -11),
        step=0.2,
        feed=1,
        exposure=int(2e3),
        imagedir="test",
        imagesize=(2456, 1842)
    ):
        self.ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
        self.debug = open("debug.txt", "w")
        self.begin = begin
        self.end = end
        self.squaresize = squaresize
        self.zlimit = zlimit
        self.zaf = zaf
        self.step = step
        self.feed = feed
        self.exposure = exposure
        self.imagedir = imagedir
        self.imagesize = imagesize
        self.bq = Queue()
        # Max queue size 10 elements (not circular queue)
        self.q = Queue(10)
        self.fin = Future()
        self.mse_capture = False
        t = Thread(
            target=self.cam,
            args=(
                self.cameraCallback,
                self,
            ),
        )
        t.start()

    @staticmethod
    def cam(callback, ctx):
        a = toupcam.Toupcam.EnumV2()
        ctx.hcam = toupcam.Toupcam.Open(a[0].id)
        ctx.hcam.put_Size(ctx.imagesize[0], ctx.imagesize[1])
        ctx.hcam.put_AutoExpoEnable(False)
        ctx.hcam.put_ExpoTime(ctx.exposure)
        ctx.hcam.put_ExpoAGain(100)
        ctx.width, ctx.height = ctx.hcam.get_Size()
        ctx.hcam.StartPullModeWithCallback(callback, ctx)
        res = ctx.fin.result()
        ctx.hcam.Stop()

    def mse_analysis(self):
        times = []
        mses = []

        # Perform autofocus before test
        self.autofocus()

        # Move away from original position by 1mm
        x, y, _, status = self.curpos()
        self.move_abs(x + 1, y, False)

        # Move back to original position
        self.move_abs(x, y, False)

        # Now start capturing photos
        self.mse_capture = True

        old = None
        for i in range(600):
            if old is None:
                old = self.bq.get()
            b = self.bq.get()
            mse = np.mean((old.buf - b.buf) ** 2)
            mses += [mse]
            times += [b.timestamp]
            old = b

        self.mse_capture = False

        fig, ax = plt.subplots()
        ax.plot(np.array(times) - times[0], mses)
        ax.set_xlabel("Time (seconds)")
        ax.set_title("MSE values across time")
        ax.set_ylabel("MSE value")
        fig.tight_layout()
        fig.savefig(f"mse_analysis.png", dpi=400)
        fig.savefig(f"mse_analysis.svg")

        np.save("mse_analysis_x.np", np.array(times))
        np.save("mse_analysis_y.np", np.array(mses))

        im = Image.fromarray(b.buf)
        im.save("mse_analysis.tif")

    def autofocus(self, step=0.001, zaf=None, fin=False):
        if zaf is None:
            zaf = self.zaf
        z0, z1 = zaf
        zstep = ((z0 - z1) / step) + 1
        allv = []

        sharpvals = []
        sharpbuf = deque(maxlen=10)
        for z in np.linspace(z0, z1, int(zstep)):
            img = self.move_abs_z(z)
            img.save(os.path.join(self.imagedir, f"af-{abs(z):.3f}.tif"), compression='tiff_deflate', tiffinfo={317: 2, 278: 1})
            # Measure sharpness from JPEG size
            buf = BytesIO()
            img.save(buf, "JPEG")
            sharpness_jpeg = len(buf.getbuffer())

            # Measure sharpness from laplacian
            r, g, b = img.split()
            sharpness_lap = laplacian(np.array(g))

            # Measure brightness
            bright = np.mean(np.array(g).flatten())
            #if bright < 35:
            #    return False

            # Add all sharpness values to list
            allv += [{"z": z, "sharp": sharpness_lap, "bright": bright}]
            sharpvals += [sharpness_lap]
            sharpbuf += [sharpness_lap]

        sharpness_best = sorted(allv, key=lambda x: x["sharp"], reverse=True)
        bestz = sharpness_best[0]["z"]
        bright = sharpness_best[0]["bright"]
        print(f"Original best sharpness {sharpness_best[0]['sharp']} at Z {sharpness_best[0]['z']}mm")

        # Move back to upper zlimit
        th = 0.07
        before = bestz + th
        self.move_abs_z(before)
        z0 = before
        z1 = bestz - th
        zstep = ((z0 - z1) / step) + 1
        nowsharp = 0
        old = None
        now = None
        for z in np.linspace(z0, z1, int(zstep)):
            now = self.move_abs_z(z)
            r, g, b = now.split()
            now_sharp = laplacian(np.array(g))
            print(now_sharp)
            if abs(now_sharp - sharpness_best[0]['sharp']) < 0.5:
                break
            old = now_sharp

        print(f"Current sharpness {now_sharp}")
        now.save(os.path.join(self.imagedir, f"af-now.tif"), compression='tiff_deflate', tiffinfo={317: 2, 278: 1})
        return True

    def wait_till_stable(self):
        count = 5
        frame = None
        old = None
        curtime = time.time()
        mses = deque(maxlen=count)
        while True:
            frame = self.q.get()
            if frame.timestamp >= curtime:
                if old is not None:
                    _, g0, _ = old.buf.split()
                    _, g1, _ = frame.buf.split()
                    mse = np.mean((np.array(g0) - np.array(g1)) ** 2)
                    mses += [mse]
                    # Note that a value of 0.01 requires lots of stability and may not converge otherwise.
                    # Can try a value of 0.1 instead.
                    if len(mses) == count and statistics.stdev(mses) < 0.1:
                        break
                old = frame
        return frame.buf

    def take_photo(self):
        photo = self.wait_till_stable()
        r, g, b = photo.split()
        now_sharp = laplacian(np.array(g))
        print(f"Current sharpness {now_sharp}")
        photo.save(os.path.join(self.imagedir, f"now.tif"), compression='tiff_deflate', tiffinfo={317: 2, 278: 1})

    def move_abs_z(self, z, mse=True, feed=None):
        z0, z1 = self.zlimit
        if z < z1 or z > z0:
            raise Exception("Sorry, can't go beyond limit")

        if feed is None:
            feed = self.feed

        self.__write(f"$J=G90 Z{z:.3f} F{feed}\n")
        dat = self.__read()

        # Check if we have reached position
        while True:
            _, _, z_, status = self.curpos()
            if status == "Idle" and math.isclose(z_, z, rel_tol=1e-3):
                break
            time.sleep(0.01)

        if mse:
            return self.wait_till_stable()
        else:
            return None

    def move_abs(self, x, y, mse=True, feed=None):
        x0, y0 = self.begin
        x1, y1 = self.end
        if not (x >= x0 and x <= x1 and y >= y0 and y <= y1):
            raise Exception("Sorry, can't go beyond limit")

        if feed is None:
            feed = self.feed

        self.__write(f"$J=G90 X{x:.3f} Y{y:.3f} F{feed}\n")
        dat = self.__read()

        # Check if we have reached position
        while True:
            x_, y_, _, status = self.curpos()
            if (
                status == "Idle"
                and math.isclose(x_, x, rel_tol=1e-4)
                and math.isclose(y_, y, rel_tol=1e-4)
            ):
                break
            time.sleep(0.01)

        if mse:
            # Check image appears stable
            return self.wait_till_stable()
        else:
            return None

    def start(self):
        sx0, sy0 = self.begin
        sx1, sy1 = self.end
        z0, z1 = self.zlimit

        done = set()
        fake = Image.new('RGB', self.imagesize)
        fake.save(os.path.join(self.imagedir, f"fake.tif"), compression='tiff_deflate', tiffinfo={317: 2, 278: 1})
        for squarey in np.arange(sy0, sy1, self.squaresize):
            for squarex in np.arange(sx0, sx1, self.squaresize):
                xpos = squarex + (self.squaresize / 2)
                ypos = squarey + (self.squaresize / 2)
                self.move_abs(xpos, ypos, mse=False)
                af = self.autofocus()
                x0, y0 = (squarex, squarey)
                x1, y1 = (squarex + self.squaresize, squarey + self.squaresize)
                ystep = ((y1 - y0) / self.step) + 1
                xstep = ((x1 - x0) / self.step) + 1
                for y in np.linspace(y0, y1, int(ystep)):
                    for x in np.linspace(x0, x1, int(xstep)):
                        if (x, y) not in done:
                            if af:
                                last = self.move_abs(x, y)
                                print(f"X{x:.3f} Y{y:.3f}")
                                # Get photos
                                last.save(os.path.join(self.imagedir, f"tile_{x:.3f}_{y:.3f}.tif"), compression='tiff_deflate', tiffinfo={317: 2, 278: 1})
                            else:
                                # Create fake image
                                os.symlink("fake.tif", os.path.join(self.imagedir, f"tile_{x:.3f}_{y:.3f}.tif"))
                            done.add((x, y))
                    tmp = x0
                    x0 = x1
                    x1 = tmp

    def stop(self):
        self.fin.set_result(True)

    def curpos(self):
        self.__write(f"?")
        dat = self.__read()
        m0 = re.match(
            r"<([a-zA-Z]+)\|MPos:(\d+\.\d+),(\d+\.\d+),(-?\d+\.\d+)\|FS:\d+,\d+(|.+)?>",
            dat,
        )
        if m0 is None:
            print(dat)
        x = float(m0.group(2))
        y = float(m0.group(3))
        z = float(m0.group(4))
        return (x, y, z, m0.group(1))

    def __read(self):
        line = self.ser.readline().decode().strip()
        self.debug.write(f"<{line}\n")
        self.debug.flush()
        return line

    def __write(self, data):
        self.ser.write(data.encode())
        self.debug.write(f">{data}\n")
        self.debug.flush()

    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == toupcam.TOUPCAM_EVENT_IMAGE:
            hcam = ctx.hcam
            bufsize = ((ctx.width * 24 + 31) // 32 * 4) * ctx.height
            buf = bytes(bufsize)
            hcam.PullImageV2(buf, 24, None)
            image = Image.frombuffer("RGB", (ctx.width, ctx.height), buf, "raw")
            r, g, b = image.split()
            if ctx.mse_capture:
                ctx.bq.put(Frame(np.array(g), ctx.width, ctx.height))
            ctx.q.put(Frame(image, ctx.width, ctx.height))

cnc = CNCMicroscope(
    (45, 45),
    (46, 46),
    squaresize=0.25, # Tried 0.5, too large, as focus off on some tiles, tried 0.25 which seems ok
    zlimit=(-10, -13),
    zaf=(-11.2, -11.5),
    step=0.1,        # Could this be increased
    feed=1,
    exposure=int(2e3),
    imagedir="test",
)

cnc.start()
cnc.stop()