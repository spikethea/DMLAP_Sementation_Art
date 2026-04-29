import os
import random
from PIL import Image

from datasets import Dataset
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer
)

# =====================================================
# CONFIG
# =====================================================

NUM_CLASSES = 4
IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 10

# Original hand-labelled data
CLEAN_IMG_DIR  = "../dataset/all/images"
CLEAN_MASK_DIR = "../dataset/all/masks"

# Auto-labelled data
AUTO_IMG_DIR  = "../unlabeled_images"
AUTO_MASK_DIR = "../auto_masks"

# Save path
OUT_DIR = "../model/retrained"

# Duplicate clean labels to give them more influence
CLEAN_REPEAT = 3

# =====================================================
# LABELS
# =====================================================

id2label = {
    0: "other",
    1: "road",
    2: "pavement",
    3: "crossing"
}

label2id = {v: k for k, v in id2label.items()}

processor = SegformerImageProcessor(size=IMAGE_SIZE)

# =====================================================
# HELPERS
# =====================================================

def list_pairs(img_dir, mask_dir):
    items = []

    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        stem = os.path.splitext(f)[0]
        mask_name = stem + ".png"

        img_path = os.path.join(img_dir, f)
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            items.append({
                "image": img_path,
                "mask": mask_path
            })

    return items


def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    mask = Image.open(example["mask"])

    encoded = processor(
        images=image,
        segmentation_maps=mask,
        return_tensors="pt"
    )

    return {
        "pixel_values": encoded["pixel_values"][0],
        "labels": encoded["labels"][0]
    }

# =====================================================
# LOAD DATA
# =====================================================

clean = list_pairs(CLEAN_IMG_DIR, CLEAN_MASK_DIR)
auto  = list_pairs(AUTO_IMG_DIR, AUTO_MASK_DIR)

# Weight clean labels more
clean = clean * CLEAN_REPEAT

all_data = clean + auto
random.shuffle(all_data)

print(f"Clean samples (weighted): {len(clean)}")
print(f"Auto samples: {len(auto)}")
print(f"Total samples: {len(all_data)}")

# Split train/val
split = int(len(all_data) * 0.9)

train_items = all_data[:split]
val_items   = all_data[split:]

train_ds = Dataset.from_list(train_items)
val_ds   = Dataset.from_list(val_items)

train_ds = train_ds.map(preprocess)
val_ds   = val_ds.map(preprocess)

# =====================================================
# MODEL
# =====================================================

# Start from your previous trained model if it exists
BASE_MODEL = "../model/final"

if os.path.exists(BASE_MODEL):
    model = SegformerForSemanticSegmentation.from_pretrained(BASE_MODEL)
else:
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

# =====================================================
# TRAINING
# =====================================================

args = TrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=3e-5,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=10,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()

# =====================================================
# SAVE FINAL MODEL
# =====================================================

model.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)

print("Retraining complete.")