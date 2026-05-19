#!/usr/bin/env python3
"""
pynq_sender_scale64.py

Send cropped 20x100x1 test sub-hitmaps to the PYNQ receiver over UDP.
Each event is flattened to 2000 uint16 words using the hls4ml/PYNQ
input scale for ap_fixed<10,4>:

    SCALE = 2^(10-4) = 64

Run on the PYNQ board, or on the host machine if UDP_IP is changed to
the PYNQ board IP address.
"""

import socket
import time

import h5py
import numpy as np


H5_PATH = "/home/xilinx/out_sub_80_180/test.h5"

# For loopback testing on the PYNQ board, use 127.0.0.1.
# If sending from a laptop/PC to the board, replace with the PYNQ IP.
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

N_EVENTS = 300
SCALE = 64
SLEEP_SECONDS = 0.10


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    with h5py.File(H5_PATH, "r") as f:
        X = f["X"]
        y = f["y"]

        n_available = X.shape[0]
        n_to_send = min(N_EVENTS, n_available)

        print("Sending UDP events")
        print("==================")
        print("H5 path       :", H5_PATH)
        print("Events to send:", n_to_send)
        print("UDP target    :", f"{UDP_IP}:{UDP_PORT}")
        print("Scale         :", SCALE)
        print()

        for i in range(n_to_send):
            event = X[i]
            label = int(y[i])

            flat_float = event.reshape(-1)
            flat_uint16 = np.round(flat_float * SCALE).astype(np.uint16)
            data = flat_uint16.tobytes()

            print(
                f"Sending event {i:3d}, true label = {label}, "
                f"bytes = {len(data)}, sum = {int(flat_uint16.sum())}, "
                f"max = {int(flat_uint16.max())}, "
                f"nonzero = {int(np.count_nonzero(flat_uint16))}"
            )

            sock.sendto(data, (UDP_IP, UDP_PORT))
            time.sleep(SLEEP_SECONDS)

    sock.close()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
