#!/usr/bin/python3

import argparse
import sys

import serial

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="CLI to jog microscope about")

    parser.add_argument(
        "--home", action=argparse.BooleanOptionalAction, help="Home CNC"
    )

    parser.add_argument(
        "--stop", action=argparse.BooleanOptionalAction, help="Stop jogging"
    )

    parser.add_argument(
        "--abs", action=argparse.BooleanOptionalAction, help="Absolute positioning"
    )

    parser.add_argument("--x", type=float, help="Jog X")
    parser.add_argument("--y", type=float, help="Jog Y")
    parser.add_argument("--z", type=float, help="Jog Z")
    parser.add_argument("--feed", type=float, default=1.0, help="Feed rate")

    args = parser.parse_args()
    ser = serial.Serial("/dev/ttyUSB0", 115200)

    x = 0
    y = 0
    z = 0

    if args.home:
        ser.write("$H\n".encode())
        ser.flush()
        sys.exit(0)

    mode = "G91"
    if args.abs:
        mode = "G90"

    data = []
    if args.x is not None:
        x = args.x
        data += [f"X{x}"]

    if args.y is not None:
        y = args.y
        data += [f"Y{y}"]

    if args.z is not None:
        z = args.z
        data += [f"Z{z}"]

    if args.stop:
        ser.write(b"\x85")
        ser.flush()
        sys.exit(0)

    if len(data) > 0:
        st = f"$J={mode} {" ".join(data)} F{args.feed}\n".encode()
        ser.write(st)
