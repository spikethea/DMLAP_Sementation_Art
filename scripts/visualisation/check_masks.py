import os
import cv2
import numpy as np

MASK_DIR = "../auto_masks"

files = sorted(os.listdir(MASK_DIR))

all_black = 0

for f in files:
    path = os.path.join(MASK_DIR, f)

    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if mask is None:
        print("Failed:", f)
        continue

    vals = np.unique(mask)

    print(f"{f}: {vals}")

    if len(vals) == 1 and vals[0] == 0:
        all_black += 1

print()
print("Total masks:", len(files))
print("All-black masks:", all_black)