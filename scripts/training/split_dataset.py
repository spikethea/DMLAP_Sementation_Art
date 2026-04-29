import os
import shutil
from sklearn.model_selection import train_test_split

IMG_DIR = "../dataset/all/images"
MASK_DIR = "../dataset/all/masks"

OUT = "../dataset"

images = sorted(os.listdir(IMG_DIR))

train, val = train_test_split(images, test_size=0.2, random_state=42)

def copy_files(file_list, split):
    os.makedirs(f"{OUT}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUT}/{split}/masks", exist_ok=True)

    for f in file_list:
        shutil.copy(f"{IMG_DIR}/{f}", f"{OUT}/{split}/images/{f}")
        shutil.copy(f"{MASK_DIR}/{f.replace('.jpg','.png')}", f"{OUT}/{split}/masks/{f.replace('.jpg','.png')}")

copy_files(train, "train")
copy_files(val, "val")