#!/usr/bin/python3

import math
import os
import queue
import re
import time
from collections import deque
from concurrent.futures import Future
from threading import Thread

import numpy as np
import serial
from PIL import Image

import toupcam


class Frame:
    def __init__(self, buf, width, height):
        self.buf = buf
        self.timestamp = time.time()
        self.width = width
        self.height = height


class CNCMicroscope:
    def __init__(self, begin, end, step, imagedir):
        self.ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
        self.begin = begin
        self.end = end
        self.step = step
        self.imagedir = imagedir
        self.q = deque(maxlen=1)
        self.fin = Future()

    @staticmethod
    def cam(callback, ctx):
        a = toupcam.Toupcam.EnumV2()
        ctx.hcam = toupcam.Toupcam.Open(a[0].id)
        ctx.hcam.put_Size(2456, 1842)
        ctx.width, ctx.height = ctx.hcam.get_Size()
        ctx.hcam.StartPullModeWithCallback(callback, ctx)
        res = ctx.fin.result()

    def start(self):
        t = Thread(
            target=self.cam,
            args=(
                self.cameraCallback,
                self,
            ),
        )
        t.start()

        x0, y0 = self.begin
        x1, y1 = self.end
        ystep = ((y1 - y0) / self.step) + 1
        xstep = ((x1 - x0) / self.step) + 1

        for y in np.linspace(y0, y1, int(ystep)):
            for x in np.linspace(x0, x1, int(xstep)):
                self.__write(f"$J=G90 X{x:.3f} Y{y:.3f} F10\n")
                dat = self.ser.readline().decode().strip()

                # Check if we have reached position
                while True:
                    self.__write(f"?")
                    dat = self.ser.readline().decode().strip()
                    m0 = re.match(
                        r"<([a-zA-Z]+)\|MPos:(\d+\.\d+),(\d+\.\d+),(-?\d+\.\d+)\|FS:\d+,\d+(|.+)?>",
                        dat,
                    )
                    x_ = float(m0.group(2))
                    y_ = float(m0.group(3))
                    if (
                        m0.group(1) == "Idle"
                        and math.isclose(x_, x, rel_tol=1e-4)
                        and math.isclose(y_, y, rel_tol=1e-4)
                    ):
                        break
                    time.sleep(0.1)

                time.sleep(2)
                print(f"X{x:.3f} Y{y:.3f}")

                # Get photos
                while True:
                    try:
                        frame = self.q.pop()
                        image = Image.frombuffer(
                            "RGB", (frame.width, frame.height), frame.buf, "raw"
                        )
                        image.save(os.path.join(self.imagedir, f"{x:.3f}_{y:.3f}.tif"))
                        break
                    except IndexError:
                        pass
            tmp = x0
            x0 = x1
            x1 = tmp
        self.fin.set_result(True)

    def __write(self, data):
        self.ser.write(data.encode())

    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == toupcam.TOUPCAM_EVENT_IMAGE:
            hcam = ctx.hcam
            bufsize = ((ctx.width * 24 + 31) // 32 * 4) * ctx.height
            buf = bytes(bufsize)
            hcam.PullImageV2(buf, 24, None)
            ctx.q.append(Frame(buf, ctx.width, ctx.height))


cnc = CNCMicroscope((0, 3), (1, 4), 0.4, "test")
cnc.start()
