import socket
import struct
import numpy as np
import cv2
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SegformerForSemanticSegmentation.from_pretrained(
    "../model/retrained"
).to(DEVICE)
model.eval()

processor = SegformerImageProcessor(size=512)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 9999))
server.listen(1)

print("Waiting for Unity...")

conn, addr = server.accept()
print("Connected:", addr)

def recvall(n):
    data = b""
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

while True:
    try:
        # 1. read frame size
        raw_len = recvall(4)
        if not raw_len:
            break

        frame_size = struct.unpack("I", raw_len)[0]

        # 2. read frame bytes
        frame_data = recvall(frame_size)
        if frame_data is None:
            break

        # 3. decode image
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = processor(images=rgb, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model(**inputs)

        mask = out.logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

        # 4. send mask (raw bytes)
        conn.sendall(mask.tobytes())

    except Exception as e:
        print("Error:", e)
        break