import cv2
import numpy as np
import sys

# -----------------------------
# INPUT
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python road_zebra_detector.py image.jpg")
    sys.exit()

img = cv2.imread(sys.argv[1])
orig = img.copy()

h, w = img.shape[:2]

# =====================================
# PART 1 — ROAD SEGMENTATION
# =====================================

# lower 2/3 of image = likely road zone
roi = img[h//3:, :]

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# asphalt tends to be low saturation / medium-dark
lower = np.array([0, 0, 40])
upper = np.array([180, 70, 170])

road = cv2.inRange(hsv, lower, upper)

# smooth noise
kernel = np.ones((9,9), np.uint8)
road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)
road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel)

# largest connected region only
contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

clean = np.zeros_like(road)

largest = None
largest_area = 0

for c in contours:
    a = cv2.contourArea(c)
    if a > largest_area:
        largest_area = a
        largest = c

if largest is not None:
    cv2.drawContours(clean, [largest], -1, 255, -1)

road = clean

# expand to full image
road_full = np.zeros((h,w), dtype=np.uint8)
road_full[h//3:, :] = road

# =====================================
# PART 2 — ZEBRA CROSSING DETECTOR
# =====================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# only inside road
white = cv2.bitwise_and(white, road_full)

contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

stripe_mask = np.zeros((h,w), dtype=np.uint8)

for c in contours:
    area = cv2.contourArea(c)
    if area < 150:
        continue

    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), ang = rect

    if rw < 3 or rh < 3:
        continue

    aspect = max(rw, rh) / min(rw, rh)

    if aspect > 1.8:
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.fillConvexPoly(stripe_mask, box, 255)

# merge stripes into one crossing object
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35,25))
zebra = cv2.morphologyEx(stripe_mask, cv2.MORPH_CLOSE, kernel)

# keep biggest crossing
contours, _ = cv2.findContours(zebra, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

clean = np.zeros_like(zebra)

largest = None
largest_area = 0

for c in contours:
    a = cv2.contourArea(c)
    if a > largest_area:
        largest_area = a
        largest = c

if largest is not None:
    cv2.drawContours(clean, [largest], -1, 255, -1)

zebra = clean

# =====================================
# PART 3 — OVERLAY
# =====================================

overlay = orig.copy()

# road = blue
blue = np.zeros_like(orig)
blue[:,:,0] = road_full

# zebra = green
green = np.zeros_like(orig)
green[:,:,1] = zebra

overlay = cv2.addWeighted(overlay,1.0,blue,0.25,0)
overlay = cv2.addWeighted(overlay,1.0,green,0.55,0)

# outlines
contours,_ = cv2.findContours(zebra, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, contours, -1, (0,255,255), 3)

# =====================================
# SAVE
# =====================================

cv2.imwrite("road_mask.png", road_full)
cv2.imwrite("zebra_mask.png", zebra)
cv2.imwrite("combined_overlay.png", overlay)

print("Saved:")
print("road_mask.png")
print("zebra_mask.png")
print("combined_overlay.png")