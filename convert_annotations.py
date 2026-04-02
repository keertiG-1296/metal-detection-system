import os
import xml.etree.ElementTree as ET

# -------------------------------------------
# CONFIG
# -------------------------------------------
DATASET_PATH = r"C:\Users\HP\Downloads\archive (1)\NEU-DET"
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

def convert_xml_to_yolo(xml_path, output_path, img_width=200, img_height=200):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get actual image size if available
    size = root.find('size')
    if size is not None:
        img_width  = int(size.find('width').text)
        img_height = int(size.find('height').text)

    lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text.strip()
        if class_name not in CLASSES:
            continue
        class_id = CLASSES.index(class_name)

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # Convert to YOLO format (normalized cx, cy, w, h)
        cx = (xmin + xmax) / 2 / img_width
        cy = (ymin + ymax) / 2 / img_height
        w  = (xmax - xmin) / img_width
        h  = (ymax - ymin) / img_height

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def convert_split(split):
    ann_dir = os.path.join(DATASET_PATH, split, "annotations")
    img_dir = os.path.join(DATASET_PATH, split, "images")
    out_dir = os.path.join(DATASET_PATH, split, "labels")
    os.makedirs(out_dir, exist_ok=True)

    xml_files = []
    for f in os.listdir(ann_dir):
        if f.endswith('.xml'):
            xml_files.append(f)

    print(f"\n Converting {split} — {len(xml_files)} XML files...")

    converted = 0
    for xml_file in xml_files:
        xml_path = os.path.join(ann_dir, xml_file)
        txt_name = xml_file.replace('.xml', '.txt')
        out_path = os.path.join(out_dir, txt_name)
        convert_xml_to_yolo(xml_path, out_path)
        converted += 1

    print(f"{converted} files converted → {out_dir}")


# Convert both splits
convert_split("train")
convert_split("validation")

# -------------------------------------------
# Create dataset.yaml for YOLOv8
# -------------------------------------------
yaml_content = f"""
path: {DATASET_PATH}
train: train/images
val: validation/images

nc: {len(CLASSES)}
names: {CLASSES}
""".strip()

yaml_path = os.path.join(DATASET_PATH, "dataset.yaml")
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\n dataset.yaml created at: {yaml_path}")
print("\ All done! Ready to train YOLOv8.")

