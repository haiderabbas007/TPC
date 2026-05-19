import socket
import numpy as np

PYNQ_IP = "192.168.1.221"
PORT = 5005

N = 2000

# Fake test event: all zeros
event = np.zeros(N, dtype=np.uint16)

packet = event.tobytes()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(packet, (PYNQ_IP, PORT))

print("Sent UDP event packet")
print("Destination:", PYNQ_IP, PORT)
print("Bytes sent:", len(packet))
