import cv2
import torch
import numpy as np
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ==========================================================
# MODEL SETUP
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SegformerForSemanticSegmentation.from_pretrained(
    "../model/retrained"
).to(DEVICE)

model.eval()

processor = SegformerImageProcessor(size=512)

cap = cv2.VideoCapture(1)

OUT_PATH = "../unity_stream/mask.png"
os.makedirs("../unity_stream", exist_ok=True)

# ==========================================================
# CONFIG
# ==========================================================
ROAD_CLASS = 1
PAVEMENT_CLASS = 2

PAVEMENT_COLOR = (128, 128, 128)
BOUNDARY_COLOR = (108, 108, 108)

GRADIENT_TOP = 108
GRADIENT_BOTTOM = 0

# zebra styling
ZEBRA_FILL = (210, 210, 210)
ZEBRA_OUTLINE = (255, 255, 255)


# ==========================================================
# ROAD GRADIENT
# ==========================================================
def make_road_gradient(mask):
    h, w = mask.shape
    road_mask = (mask == ROAD_CLASS)

    rows = np.where(road_mask.any(axis=1))[0]

    if len(rows) == 0:
        return np.zeros((h, w), dtype=np.float32)

    top = float(rows.min())
    bottom = float(rows.max())
    span = bottom - top if bottom > top else 1.0

    ys = np.arange(h, dtype=np.float32)
    t = np.clip((ys - top) / span, 0.0, 1.0)

    t_map = np.tile(t[:, None], (1, w))
    t_map[~road_mask] = 0.0

    return t_map

# ==========================================================
# PAVEMENT GRADIENT
# ==========================================================

def get_pavement_islands(mask):
    pav = (mask == PAVEMENT_CLASS).astype(np.uint8)
    num, labels = cv2.connectedComponents(pav)
    return num, labels, pav

def pavement_extrusion_contours(mask, pavement_mask):
    pav = (mask == PAVEMENT_CLASS).astype(np.uint8)

    contours, _ = cv2.findContours(
        pav,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    H, W = mask.shape
    depth = np.zeros((H, W, 3), dtype=np.float32)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 80:
            continue

        # base shape
        base = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(base, c, 1)

        color = 180  # top brightness

        num_layers = 12
        thickness = 25
        start = (num_layers - 1) * thickness

        for i in range(num_layers):

            shift = start - i * thickness

            offset = np.roll(base, shift, axis=0)

            if shift < 0:
                offset[shift:, :] = 0
            elif shift > 0:
                offset[:shift, :] = 0

            shade = max(20, color - i * 10)

            mask = offset.astype(bool) & (~pavement_mask)
            depth[mask] = [shade, shade, shade]

    return depth
# ==========================================================
# HARD-CODED ZEBRA DETECTOR
# ==========================================================
def detect_zebra(frame, road_mask):
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    road_u8 = (road_mask.astype(np.uint8)) * 255
    white = cv2.bitwise_and(white, road_u8)

    contours, _ = cv2.findContours(
        white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    stripe_mask = np.zeros((h, w), dtype=np.uint8)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 150:
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), _ = rect

        if rw < 3 or rh < 3:
            continue

        aspect = max(rw, rh) / max(min(rw, rh), 1)

        if aspect > 1.8:
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.fillConvexPoly(stripe_mask, box, 255)

    # merge stripes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 25))
    zebra = cv2.morphologyEx(stripe_mask, cv2.MORPH_CLOSE, kernel)

    # keep biggest crossing
    contours, _ = cv2.findContours(
        zebra, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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

    return clean


# ==========================================================
# MAIN EFFECT
# ==========================================================
def apply_flat_segmentation(frame, mask):
    output = frame.copy().astype(np.float32)

    road_mask = (mask == ROAD_CLASS)
    H, W = mask.shape

    pavement_mask = (mask == PAVEMENT_CLASS)

    # ------------------------------
    # Road gradient
    # ------------------------------
    if road_mask.any():
        t_map = make_road_gradient(mask)

        grad = GRADIENT_TOP + (
            GRADIENT_BOTTOM - GRADIENT_TOP
        ) * t_map

        grad_bgr = grad[..., None]

        output[road_mask] = grad_bgr[road_mask]

    # ------------------------------
    # Boundary
    # ------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    road_d = cv2.dilate(
        road_mask.astype(np.uint8), kernel, iterations=2
    )

    pav_d = cv2.dilate(
        pavement_mask.astype(np.uint8), kernel, iterations=2
    )

    boundary = (road_d & pav_d).astype(bool)

    output[boundary] = BOUNDARY_COLOR
#=========================================================
    #PAVEMENT CONTOUR
#=========================================================
    depth = pavement_extrusion_contours(mask, pavement_mask)



    depth_2d = depth[..., 0] if depth.ndim == 3 else depth

    extrusion = depth_2d > 0
    extrusion_only = extrusion & (~pavement_mask)

    val = depth_2d[extrusion_only][:, None]
    output[extrusion_only] = np.repeat(val, 3, axis=1)

    # ==================================================
    # ZEBRA CROSSING OVERLAY
    # ==================================================
    zebra_raw = detect_zebra(frame, road_mask)

    # --- convert raw zebra stripes into one convex bridge shape ---
    contours, _ = cv2.findContours(
        zebra_raw,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    zebra = np.zeros_like(zebra_raw)

    if len(contours) > 0:
        pts = np.concatenate([c.reshape(-1,2) for c in contours if len(c) > 2], axis=0)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(zebra, hull, 255)

    zebra_bool = zebra > 0
    # --------------------------------------------------
    # 1. REMOVE ANY ZEBRA ON TOP OF PAVEMENT
    # keeps crossing only on road / under pavement edges
    # --------------------------------------------------
    zebra_bool[pavement_mask] = False

    # --------------------------------------------------
    # 2. EXTRUDED UNDERSIDE SHADOW / DEPTH
    # duplicate zebra shape slightly downward
    # --------------------------------------------------
    shadow = np.zeros_like(zebra)
    shadow[10:, :] = zebra[:-10, :]   # push downward
    # shadow[:10, :] = 0                                     # clear wraparound

    # keep shadow only on road, not pavement
    shadow[pavement_mask] = 0

    shadow_bool = shadow > 0

    # darker underside block
    output[shadow_bool] = (88, 88, 88)

    # optional blur edge for softer depth
    shadow_blur = cv2.GaussianBlur(shadow, (9,9), 0)
    shadow_soft = shadow_blur > 25
    output[shadow_soft] = np.minimum(output[shadow_soft], 70)

    # --------------------------------------------------
    # 3. TOP SURFACE OF CROSSING
    # --------------------------------------------------
    output[zebra_bool] = ZEBRA_FILL

    # --------------------------------------------------
    # 4. FRONT FACE / SIDE WALLS
    # connect top to shadow for pseudo extrusion
    # --------------------------------------------------
    for dy in range(1, 10):
        wall = np.roll(zebra.astype(np.uint8), dy, axis=0)
        wall[:dy, :] = 0
        wall[pavement_mask] = 0

        wall_only = (wall > 0) & (~zebra_bool)

        shade = 90 - dy * 4
        output[wall_only] = (shade, shade, shade)

    # --------------------------------------------------
    # 5. OUTLINE TOP SURFACE
    # --------------------------------------------------
    contours, _ = cv2.findContours(
        zebra.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # OUTLINE (optional overlay only)
    output_u8 = output.astype(np.uint8)

    cv2.drawContours(
        output_u8,
        contours,
        -1,
        (255, 255, 255),
        2
    )

    # 1. pavement on top of zebra, SO THAT SUBTRACT WORKS
    output[pavement_mask] = (128, 128, 128)

    return output_u8


# ==========================================================
# LIVE LOOP
# ==========================================================
while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    inputs = processor(
        images=rgb,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    upsampled = torch.nn.functional.interpolate(
        outputs.logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    mask = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    result = apply_flat_segmentation(frame, mask)

    cv2.imwrite(OUT_PATH, mask)

    cv2.imshow("camera", frame)
    cv2.imshow("mask", mask * 80)
    cv2.imshow("segmentation", result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()