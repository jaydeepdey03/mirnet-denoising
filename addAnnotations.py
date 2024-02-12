import os
import csv
from PIL import Image, ImageDraw
from tqdm import tqdm  # Import tqdm for the progress bar

def read_csv_file(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        annotations = {}

        for row in reader:
            image_name, _, _, class_label, x_min, y_min, x_max, y_max = row
            box = list(map(int, [x_min, y_min, x_max, y_max]))

            if image_name not in annotations:
                annotations[image_name] = {'class_labels': [], 'boxes': []}

            annotations[image_name]['class_labels'].append(class_label)
            annotations[image_name]['boxes'].append(box)

    return annotations

def annotate_images(csv_path, image_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read annotations from CSV
    annotations = read_csv_file(csv_path)

    # Annotate images in the specified directory
    for image_name, data in tqdm(annotations.items(), desc='Annotating images', unit='image'):
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, "annotated_" + image_name)
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        for class_label, box in zip(data['class_labels'], data['boxes']):
            x_min, y_min, x_max, y_max = box

            # Draw bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

            # Draw class label underneath the bounding box
            text = f"{class_label}"
            _, _, width, text_height = draw.textbbox((0, 0), text=text)
            draw.rectangle([x_min, y_max, x_max, y_max + text_height + 4], fill="green")
            draw.text((x_min, y_max), text, fill="white")

        img.save(output_path)

# Example usage
base_dir = "CNN/MIRNet-Keras"
csv_path = os.path.join(base_dir, "dataset_polarimetric/train/labels.csv")
image_dir = os.path.join(base_dir, "dataset_polarimetric/train/PARAM_POLAR")
output_dir = os.path.join(base_dir, "annotated_images")

annotate_images(csv_path, image_dir, output_dir)
