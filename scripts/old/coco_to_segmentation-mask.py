import json
import numpy as np
import cv2
import os
from tqdm import tqdm

COCO_PATH = "../coco/annotations.json"
IMG_DIR = "../coco/images"
OUT_IMG_DIR = "../dataset/all/images"
OUT_MASK_DIR = "../dataset/all/masks"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# -----------------------------------
# CLASS ID MAP (your training labels)
# -----------------------------------
cat_map = {
    2: 0,  # other
    1: 1,  # road
    4: 2,  # pavement
    3: 3   # crossing
}

# -----------------------------------
# DRAW PRIORITY
# higher number = drawn later = wins overlap
# -----------------------------------
priority = {
    2: 1,  # other
    1: 2,  # road
    4: 3,  # pavement
    3: 4   # crossing (highest)
}

with open(COCO_PATH) as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

# -----------------------------------
# Group annotations by image first
# (much faster than scanning all anns each loop)
# -----------------------------------
anns_by_image = {}

for ann in coco["annotations"]:
    img_id = ann["image_id"]
    anns_by_image.setdefault(img_id, []).append(ann)

# -----------------------------------
# PROCESS EACH IMAGE
# -----------------------------------
for img_id, img_info in tqdm(images.items()):

    h = img_info["height"]
    w = img_info["width"]

    mask = np.zeros((h, w), dtype=np.uint8)

    # get annotations for this image only
    anns = anns_by_image.get(img_id, [])

    # sort by priority so highest gets drawn last
    anns.sort(key=lambda a: priority.get(a["category_id"], 0))

    for ann in anns:

        cat_id = ann["category_id"]

        if cat_id not in cat_map:
            continue

        for seg in ann["segmentation"]:

            pts = np.array(seg).reshape(-1, 2).astype(np.int32)

            cv2.fillPoly(
                mask,
                [pts],
                cat_map[cat_id]
            )

    # -----------------------------------
    # Copy image
    # -----------------------------------
    img_src = os.path.join(IMG_DIR, img_info["file_name"])
    img_dst = os.path.join(OUT_IMG_DIR, img_info["file_name"])

    img = cv2.imread(img_src)
    cv2.imwrite(img_dst, img)

    # -----------------------------------
    # Save mask
    # -----------------------------------
    mask_name = os.path.splitext(img_info["file_name"])[0] + ".png"

    cv2.imwrite(
        os.path.join(OUT_MASK_DIR, mask_name),
        mask
    )