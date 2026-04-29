import sys
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# --------------------------------------------------------
# SETTINGS
# --------------------------------------------------------
ROAD_CLASS = 6      # ADE20K road label
BOTTOM_MARGIN = 5   # px for touching bottom

# --------------------------------------------------------
# INPUT
# --------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python segformer_road_segmentation_v2.py image.jpg")
    sys.exit()

image_path = sys.argv[1]

img_bgr = cv2.imread(image_path)
if img_bgr is None:
    print("Could not load image.")
    sys.exit()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
pil = Image.fromarray(img_rgb)

h, w = img_bgr.shape[:2]

# --------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------
print("Loading model...")

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

model.eval()

# --------------------------------------------------------
# RUN MODEL
# --------------------------------------------------------
print("Running segmentation...")

inputs = processor(images=pil, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

upsampled = torch.nn.functional.interpolate(
    logits,
    size=(h, w),
    mode="bilinear",
    align_corners=False
)

pred = upsampled.argmax(dim=1)[0].cpu().numpy()

road_mask = np.uint8(pred == ROAD_CLASS) * 255

# --------------------------------------------------------
# STEP 1: Keep only bottom-connected road blobs
# --------------------------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    road_mask, connectivity=8
)

clean = np.zeros_like(road_mask)

for i in range(1, num_labels):
    x, y, ww, hh, area = stats[i]

    touches_bottom = (y + hh >= h - BOTTOM_MARGIN)

    if touches_bottom and area > 500:
        clean[labels == i] = 255

road_mask = clean

# --------------------------------------------------------
# STEP 2: Perspective trapezoid crop
# Keeps plausible drivable region only
# --------------------------------------------------------
roi = np.zeros_like(road_mask)

pts = np.array([[
    (int(w * 0.05), h),
    (int(w * 0.38), int(h * 0.48)),
    (int(w * 0.62), int(h * 0.48)),
    (int(w * 0.95), h)
]], dtype=np.int32)

cv2.fillPoly(roi, pts, 255)

road_mask = cv2.bitwise_and(road_mask, roi)

# --------------------------------------------------------
# STEP 3: Lower-image weighting
# suppress upper weak detections
# --------------------------------------------------------
fade = np.linspace(0.2, 1.0, h).reshape(h,1)
weighted = (road_mask.astype(np.float32) * fade).astype(np.uint8)
road_mask = np.where(weighted > 100, 255, 0).astype(np.uint8)

# --------------------------------------------------------
# STEP 4: Morphology cleanup
# --------------------------------------------------------
kernel = np.ones((9,9), np.uint8)

road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

# --------------------------------------------------------
# STEP 5: Fill holes
# --------------------------------------------------------
contours, _ = cv2.findContours(
    road_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

filled = np.zeros_like(road_mask)
cv2.drawContours(filled, contours, -1, 255, -1)

road_mask = filled

# --------------------------------------------------------
# OVERLAY
# --------------------------------------------------------
overlay = img_bgr.copy()

blue = np.zeros_like(img_bgr)
blue[:,:,0] = road_mask

overlay = cv2.addWeighted(overlay, 1.0, blue, 0.45, 0)

# contour edge
contours, _ = cv2.findContours(
    road_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(overlay, contours, -1, (0,255,255), 2)

# --------------------------------------------------------
# SAVE
# --------------------------------------------------------
cv2.imwrite("road_mask.png", road_mask)
cv2.imwrite("segmented_overlay.png", overlay)

print("Saved:")
print("road_mask.png")
print("segmented_overlay.png")