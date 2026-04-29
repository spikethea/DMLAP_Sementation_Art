import cv2
import torch
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = SegformerImageProcessor(size=512)

# LOAD TRAINED MODEL FOLDER
model = SegformerForSemanticSegmentation.from_pretrained(
    "../model/retrained"
)

model.to(DEVICE)
model.eval()

IN_DIR = "../unlabeled_images"
OUT_DIR = "../retrained_masks"

os.makedirs(OUT_DIR, exist_ok=True)

for f in os.listdir(IN_DIR):

    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(IN_DIR, f)

    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    inputs = processor(images=rgb, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # resize prediction to original image size
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    pred = upsampled.argmax(dim=1)[0].cpu().numpy()

    out_name = os.path.splitext(f)[0] + ".png"

    cv2.imwrite(
        os.path.join(OUT_DIR, out_name),
        pred.astype("uint8")
    )

print("Done.")