"""
train.py - PCB Defect Detection Training Script
Author: Madusha Nadiranga
Project: Deep Learning-Based Visual Quality Inspection of PCBs
"""

import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# ====================================================
# 1. Dataset Preparation
# ====================================================
def prepare_dataset(dataset_path, test_size=0.3, val_size=0.33):
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")

    all_images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    # Split dataset: train (70%), val (20%), test (10%)
    train_imgs, temp_imgs = train_test_split(all_images, test_size=test_size, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=val_size, random_state=42)

    def move_files(file_list, split):
        os.makedirs(os.path.join(dataset_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, split, "labels"), exist_ok=True)
        for f in file_list:
            shutil.copy(os.path.join(images_dir, f), os.path.join(dataset_path, split, "images", f))
            label_file = f.replace(".jpg", ".txt")
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(dataset_path, split, "labels", label_file))

    move_files(train_imgs, "train")
    move_files(val_imgs, "val")
    move_files(test_imgs, "test")

    print(f"✅ Dataset prepared: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")


# ====================================================
# 2. Create YAML config
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
    yaml_file = os.path.join(dataset_path, "data.yaml")
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    print(f"✅ YAML config file created at {yaml_file}")
    return yaml_file


# ====================================================
# 3. Train YOLO Model
# ====================================================
def train_model(yaml_file, model_name="yolov8s.pt", epochs=50, imgsz=640, batch=16, device=0):
    model = YOLO(model_name)
    results = model.train(
        data=yaml_file,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=10,
        device=device
    )
    print("✅ Training completed!")
    return model, results


# ====================================================
# 4. Validate Model
# ====================================================
def validate_model(model):
    metrics = model.val()
    print("✅ Validation completed!")
    print(metrics)
    return metrics


# ====================================================
# 5. Run Predictions
# ====================================================
def predict_images(model, source_dir="data/test/images"):
    preds = model.predict(source=source_dir, save=True)
    print(f"✅ Predictions saved at: {preds[0].save_dir}")


# ====================================================
# 6. Main
# ====================================================
if __name__ == "__main__":
    dataset_path = "data"
    num_classes = 6
    class_names = ['missing_component', 'solder_bridge', 'misalignment', 'scratch', 'short_circuit', 'other']

    # Step 1: Dataset preparation
    prepare_dataset(dataset_path)

    # Step 2: YAML config
    yaml_file = create_yaml(dataset_path, num_classes, class_names)

    # Step 3: Train model
    model, _ = train_model(yaml_file, epochs=50, imgsz=640, batch=16, device=0)

    # Step 4: Validate model
    validate_model(model)

    # Step 5: Predictions
    predict_images(model)
