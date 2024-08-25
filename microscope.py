#!/usr/bin/python3

import math
import os
import queue
import re
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
        zlimit=(-1, -11),
        step=0.2,
        feed=10,
        exposure=int(2e3),
        imagedir="test",
    ):
        self.ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
        self.begin = begin
        self.end = end
        self.zlimit = zlimit
        self.step = step
        self.feed = feed
        self.exposure = exposure
        self.imagedir = imagedir
        self.bq = Queue()
        self.q = deque(maxlen=10)
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
        ctx.hcam.put_Size(2456, 1842)
        ctx.hcam.put_AutoExpoEnable(False)
        ctx.hcam.put_ExpoTime(ctx.exposure)
        ctx.hcam.put_ExpoAGain(100)
        ctx.width, ctx.height = ctx.hcam.get_Size()
        ctx.hcam.StartPullModeWithCallback(callback, ctx)
        res = ctx.fin.result()
        ctx.hcam.Stop()

    def mse_analysis(self):
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
            old = b

        self.mse_capture = False

        fig, ax = plt.subplots()
        ax.plot(np.arange(len(mses)), mses)
        ax.set_xlabel("Nth value")
        ax.set_title("MSE values")
        ax.set_ylabel("MSE")
        fig.tight_layout()
        fig.savefig(f"mse_analysis.png", dpi=400)
        fig.savefig(f"mse_analysis.svg")

        im = Image.fromarray(b.buf)
        im.save("mse_analysis.tif")

    def autofocus(self, step=0.01, zlimit=None, fin=False):
        if zlimit is None:
            zlimit = self.zlimit
        z0, z1 = zlimit
        zstep = ((z0 - z1) / step) + 1
        allv = []

        for z in np.linspace(z0, z1, int(zstep)):
            img = self.move_abs_z(z)
            img.save(os.path.join(self.imagedir, f"{abs(z):.3f}.tif"))

            # Measure sharpness from JPEG size
            buf = BytesIO()
            img.save(buf, "JPEG")
            sharpness_jpeg = len(buf.getbuffer())

            # Measure sharpness from laplacian
            r, g, b = img.split()
            sharpness_lap = laplacian(np.array(g))

            # Add all sharpness values to list
            allv += [{"z": z, "sharp": sharpness_lap}]

        sharpness_best = sorted(allv, key=lambda x: x["sharp"], reverse=True)
        bestz = sharpness_best[0]["z"]

        print(bestz)

        # Move back to upper zlimit (necessary to avoid backlash - if you don't do this, next 'move' isn't correct)
        self.move_abs_z(z0)
        self.move_abs_z(bestz)

    def move_abs_z(self, z, mse=True):
        self.__write(f"$J=G90 Z{z:.3f} F{self.feed}\n")
        dat = self.__read()

        # Check if we have reached position
        while True:
            _, _, z_, status = self.curpos()
            if status == "Idle" and math.isclose(z_, z, rel_tol=1e-3):
                break

        if mse:
            time.sleep(0.5)
            # Check image appears stable
            last = None
            mse = 100
            while mse > 26:
                if len(self.q) >= 10:
                    last = list(self.q)[-2:]
                    mse = np.mean((np.array(last[0].buf) - np.array(last[1].buf)) ** 2)
                    #print(f"mse {mse}")
                    last = last[1].buf
            return last
        else:
            return None

    def move_abs(self, x, y, mse=True):
        self.__write(f"$J=G90 X{x:.3f} Y{y:.3f} F{self.feed}\n")
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

        if mse:
            # Check image appears stable
            last = None
            mse = 100
            while mse > 25:
                if len(self.q) >= 10:
                    last = list(self.q)[-2:]
                    mse = np.mean((np.array(last[0].buf) - np.array(last[1].buf)) ** 2)
                    print(f"mse {mse}")
                    last = last[1].buf
            return last
        else:
            return None

    def start(self):
        x0, y0 = self.begin
        x1, y1 = self.end
        ystep = ((y1 - y0) / self.step) + 1
        xstep = ((x1 - x0) / self.step) + 1

        for y in np.linspace(y0, y1, int(ystep)):
            for x in np.linspace(x0, x1, int(xstep)):
                last = self.move_abs(x, y)
                print(f"X{x:.3f} Y{y:.3f}")

                # Get photos
                last.save(os.path.join(self.imagedir, f"{x:.3f}_{y:.3f}.tif"))

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
        return self.ser.readline().decode().strip()

    def __write(self, data):
        self.ser.write(data.encode())

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
            ctx.q.append(Frame(image, ctx.width, ctx.height))


cnc = CNCMicroscope(
    (20, 70),
    (20.5, 70.5),
    zlimit=(-10, -12),
    step=0.2,
    feed=10,
    exposure=int(2e3),
    imagedir="test",
)

cnc.autofocus()
cnc.stop()

# cnc.mse_analysis()
# cnc.start()
