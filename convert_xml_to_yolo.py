
import os
import xml.etree.ElementTree as ET

# Define your class mapping here
class_mapping = {
    "missing_hole": 0,
    "mouse_bite": 1,
    "open_circuit": 2,
    "short": 3,
    "spur": 4,
    "spurious_copper": 5
}

# Paths
annotation_root = "data/Annotations"  # Change this to your XML folder
output_label_dir = "data/YOLO_Labels"
os.makedirs(output_label_dir, exist_ok=True)

# Loop through all subfolders and XML files
for defect_type in os.listdir(annotation_root):
    folder_path = os.path.join(annotation_root, defect_type)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if not file.endswith(".xml"):
            continue

        xml_path = os.path.join(folder_path, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_width = int(root.find("size/width").text)
        image_height = int(root.find("size/height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip().lower()
            if class_name not in class_mapping:
                continue

            class_id = class_mapping[class_name]
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Write YOLO label file
        txt_filename = os.path.splitext(file)[0] + ".txt"
        with open(os.path.join(output_label_dir, txt_filename), "w") as f:
            f.write("\n".join(yolo_lines))
