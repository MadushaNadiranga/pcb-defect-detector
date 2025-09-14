"""
preprocess_and_split.py - PCB Dataset Preparation Script
Author: Madusha Nadiranga
Project: Deep Learning-Based Visual Quality Inspection of PCBs
"""

import os
import shutil
from sklearn.model_selection import train_test_split


# ====================================================
# 1. Dataset Splitting Function
# ====================================================
def prepare_dataset(dataset_path, test_size=0.3, val_size=0.33):

    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")

    all_images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    # Split: Train (70%), Val (20%), Test (10%)
    train_imgs, temp_imgs = train_test_split(all_images, test_size=test_size, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=val_size, random_state=42)

    def move_files(file_list, split):
        os.makedirs(os.path.join(dataset_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, split, "labels"), exist_ok=True)
        for f in file_list:
            shutil.copy(os.path.join(images_dir, f), os.path.join(dataset_path, split, "images", f))
            label_file = f.replace(".jpg", ".txt")
            if os.path.exists(os.path.join(labels_dir, label_file)):
                shutil.copy(os.path.join(labels_dir, label_file), os.path.join(dataset_path, split, "labels", label_file))

    move_files(train_imgs, "train")
    move_files(val_imgs, "val")
    move_files(test_imgs, "test")
    print(f"Dataset prepared: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")


# ====================================================
# 2. YAML Creation Function
# ====================================================
def create_yaml(dataset_path, num_classes, class_names):

    yaml_content = f"""
path: {dataset_path}
train: train/images
val: val/images
test: test/images

nc: {num_classes}
names: {class_names}
"""
    yaml_file = os.path.join(dataset_path, "pcb.yaml")
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    print(f"✅ YAML config file created at {yaml_file}")
    return yaml_file



