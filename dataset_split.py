import os
import shutil
import random
import warnings
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

# ===============================
# 1. Dataset Preparation
# ===============================
source_mask = "data/with_mask"
source_nomask = "data/without_mask"

base_dir = "mask_dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

for dir_path in [train_dir, test_dir]:
    os.makedirs(os.path.join(dir_path, "with_mask"), exist_ok=True)
    os.makedirs(os.path.join(dir_path, "without_mask"), exist_ok=True)

def split_data(source, train_target, test_target, split_ratio=0.8):
    files = os.listdir(source)
    random.shuffle(files)
    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    test_files = files[split_index:]

    for f in train_files:
        shutil.copy(os.path.join(source, f), train_target)
    for f in test_files:
        shutil.copy(os.path.join(source, f), test_target)

split_data(source_mask, os.path.join(train_dir, "with_mask"), os.path.join(test_dir, "with_mask"))
split_data(source_nomask, os.path.join(train_dir, "without_mask"), os.path.join(test_dir, "without_mask"))

print("Dataset split into train/test folders successfully!")




