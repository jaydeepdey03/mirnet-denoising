import os
import pandas as pd

# Define class mapping
class_mapping = {'car': 0, 'person': 1, 'bike': 2}

def convert_coordinates_to_yolo(xmin, ymin, xmax, ymax, width, height):
    x_center = (xmin + xmax) / (2 * width)
    y_center = (ymin + ymax) / (2 * height)
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height

def create_yolo_text_file(image_id, labels, output_folder):
    with open(os.path.join(output_folder, f"{image_id}.txt"), 'w') as file:
        for _, label in labels.iterrows():
            class_id = class_mapping[label['class']]
            x_center, y_center, box_width, box_height = convert_coordinates_to_yolo(
                label['xmin'], label['ymin'], label['xmax'], label['ymax'], label['width'], label['height']
            )
            file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

def convert_csv_to_yolo_labels(csv_path, output_folder):
    # Read labels from CSV
    df_labels = pd.read_csv(csv_path)

    # Iterate through each image
    for image_id, labels in df_labels.groupby('filename'):
        create_yolo_text_file(image_id.split('.')[0], labels, output_folder)

# Example usage
csv_path = 'CNN/MIRNet-Keras/dataset_polarimetric/validation/labels.csv'
output_folder = 'CNN/MIRNet-Keras/dataset_polarimetric/validation/text_files'

convert_csv_to_yolo_labels(csv_path, output_folder)
