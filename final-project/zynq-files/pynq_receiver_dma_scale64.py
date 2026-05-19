#!/usr/bin/env python3
"""
pynq_receiver_dma_scale64.py

PYNQ receiver for the CNN-DMA FPGA validation.

This script:
  1. Loads the CNN-DMA bitstream.
  2. Allocates input/output DMA buffers.
  3. Opens a UDP socket on port 5005.
  4. Receives one 20x100 event per packet as 2000 uint16 words.
  5. Sends the event through AXI DMA to the FPGA CNN accelerator.
  6. Reads the output word, decodes trigger and score fields.
  7. Compares the trigger to labels from test.h5.
  8. Writes a CSV log for all processed events.

Expected UDP packet size:
    2000 uint16 words = 4000 bytes
"""

from pathlib import Path
import csv
import socket
import time

import h5py
import numpy as np
from pynq import Overlay, allocate


BITSTREAM_PATH = "/home/xilinx/jupyter_notebooks/cnn_dma.bit"
LABEL_PATH = "/home/xilinx/out_sub_80_180/test.h5"
RESULT_CSV = "/home/xilinx/fpga_300_results_scale64.csv"

PORT = 5005
N = 2000
N_EVENTS = 300


def decode_result(word: int):
    """Decode the 32-bit FPGA output word."""
    trigger = (word >> 31) & 0x1

    score_raw = word & 0xFFFF
    score_signed = score_raw
    if score_signed >= 2**15:
        score_signed -= 2**16

    return trigger, score_raw, score_signed


def main() -> None:
    print("Loading overlay:", BITSTREAM_PATH)
    ol = Overlay(BITSTREAM_PATH)
    ol.download()
    dma = ol.axi_dma_0

    print("Allocating DMA buffers")
    event_buffer = allocate(shape=(N,), dtype=np.uint16)
    result_buffer = allocate(shape=(1,), dtype=np.uint32)

    with h5py.File(LABEL_PATH, "r") as f:
        labels = f["y"][:N_EVENTS].astype(int)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))

    print("Listening for UDP event packets on port", PORT)
    print("PYNQ is ready...")
    print()

    csv_path = Path(RESULT_CSV)
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)

    writer.writerow([
        "event_idx",
        "true_label",
        "prediction",
        "correct",
        "trigger",
        "score_raw",
        "score_signed",
        "raw_word_hex",
        "event_sum",
        "event_max",
        "event_nonzero",
        "latency_ms",
    ])

    event_idx = 0
    correct_count = 0
    latencies = []

    try:
        while event_idx < N_EVENTS:
            data, addr = sock.recvfrom(N * 2)

            if len(data) != N * 2:
                print(
                    "Bad packet from",
                    addr,
                    "bytes =",
                    len(data),
                    "expected =",
                    N * 2,
                )
                continue

            event_np = np.frombuffer(data, dtype=np.uint16)

            event_buffer[:] = event_np
            result_buffer[:] = 0

            t0 = time.time()

            # Start receive first, then send.
            dma.recvchannel.transfer(result_buffer)
            dma.sendchannel.transfer(event_buffer)

            dma.sendchannel.wait()
            dma.recvchannel.wait()

            t1 = time.time()

            word = int(result_buffer[0])
            trigger, score_raw, score_signed = decode_result(word)

            latency_ms = (t1 - t0) * 1000.0
            latencies.append(latency_ms)

            true_label = int(labels[event_idx])
            prediction = int(trigger)
            correct = int(prediction == true_label)
            correct_count += correct

            writer.writerow([
                event_idx,
                true_label,
                prediction,
                correct,
                trigger,
                score_raw,
                score_signed,
                hex(word),
                int(event_np.sum()),
                int(event_np.max()),
                int(np.count_nonzero(event_np)),
                latency_ms,
            ])
            csv_file.flush()

            print("Packet from:", addr)
            print("event idx    =", event_idx)
            print("true label   =", true_label)
            print("prediction   =", prediction)
            print("correct      =", correct)
            print("event sum    =", int(event_np.sum()))
            print("event max    =", int(event_np.max()))
            print("first 20     =", event_np[:20])
            print("raw word     =", hex(word))
            print("trigger      =", trigger)
            print("score_raw    =", score_raw)
            print("score_signed =", score_signed)
            print("latency ms   =", latency_ms)
            print("-" * 50)

            event_idx += 1

    except KeyboardInterrupt:
        print()
        print("Interrupted by user.")

    finally:
        csv_file.close()
        sock.close()

    latencies_np = np.array(latencies, dtype=float)

    print()
    print("================ FINAL RESULT ================")
    print("Events received =", event_idx)

    if event_idx > 0:
        print("Correct         =", f"{correct_count}/{event_idx}")
        print("Accuracy        =", f"{correct_count / event_idx:.4f}")

    if len(latencies_np) > 0:
        print("Avg latency ms  =", f"{latencies_np.mean():.4f}")
        print("Min latency ms  =", f"{latencies_np.min():.4f}")
        print("Max latency ms  =", f"{latencies_np.max():.4f}")
        print("Std latency ms  =", f"{latencies_np.std():.4f}")

    print("Saved CSV       =", RESULT_CSV)
    print("==============================================")


if __name__ == "__main__":
    main()
