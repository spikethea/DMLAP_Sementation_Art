import os
import cv2
import numpy as np
import random

IMG_DIR = "../unlabeled_images"
MASK_DIR = "../auto_masks"
OUT_DIR = "../mask_previews"

os.makedirs(OUT_DIR, exist_ok=True)

# BGR colors for OpenCV
COLORS = {
    1: (0, 0, 255),    # road = red
    2: (0, 255, 0),    # pavement = green
    3: (255, 0, 0),    # crossing = blue
}

files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(files)

# preview first 50 random images
for f in files[:50]:
    stem = os.path.splitext(f)[0]

    img_path = os.path.join(IMG_DIR, f)
    mask_path = os.path.join(MASK_DIR, stem + ".png")

    if not os.path.exists(mask_path):
        continue

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    overlay = img.copy()

    for cls, color in COLORS.items():
        overlay[mask == cls] = color

    result = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)

    cv2.imwrite(os.path.join(OUT_DIR, stem + "_preview.jpg"), result)

print("Done.")