import os
from PIL import Image
from datasets import Dataset
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer
)

IMAGE_SIZE = 512
NUM_CLASSES = 4
DEVICE_BATCH = 4

TRAIN_IMG = "../dataset/train/images"
TRAIN_MASK = "../dataset/train/masks"

VAL_IMG = "../dataset/val/images"
VAL_MASK = "../dataset/val/masks"

id2label = {
    0:"other",
    1:"road",
    2:"pavement",
    3:"crossing"
}

label2id = {v:k for k,v in id2label.items()}

processor = SegformerImageProcessor(size=IMAGE_SIZE)

def make_dataset(img_dir, mask_dir):
    files = sorted(os.listdir(img_dir))

    return Dataset.from_dict({
        "image":[os.path.join(img_dir,f) for f in files],
        "mask":[os.path.join(mask_dir,f.replace(".jpg",".png")) for f in files]
    })

train_ds = make_dataset(TRAIN_IMG, TRAIN_MASK)
val_ds   = make_dataset(VAL_IMG, VAL_MASK)

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

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

model = SegformerForSemanticSegmentation.from_pretrained(
    "../model/final"
)

args = TrainingArguments(
    output_dir="../model/checkpoints",
    learning_rate=6e-5,
    num_train_epochs=20,
    per_device_train_batch_size=DEVICE_BATCH,
    per_device_eval_batch_size=DEVICE_BATCH,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=5,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()

model.save_pretrained("../model/final")