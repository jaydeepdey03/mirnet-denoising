import os
import csv
from PIL import Image, ImageDraw

def read_csv_file(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        annotations = [row[:3] + row[3:] for row in reader]

    return annotations

def annotate_images(csv_path, image_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read annotations from CSV
    annotations = read_csv_file(csv_path)
    
    # Annotate images in the specified directory
    for annotation in annotations:
        image_name, class_label = annotation[0], annotation[3]
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, "annotated_" + image_name)
        
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        x_min, y_min, x_max, y_max = map(int, annotation[4:])
        
        # Draw bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        
        # Draw class label underneath the bounding box
        text = f"Class: {class_label}"
        text_width, text_height = draw.textsize(text)
        draw.rectangle([x_min, y_max, x_max, y_max + text_height + 4], fill="green")
        draw.text((x_min, y_max), text, fill="white")
        
        img.save(output_path)

# Example usage
csv_path = "/Users/jaydeepdey/Desktop/capstone code/CNN_Denoising/MIRNet-Keras/dataset_polarimetric/train/labels.csv"
image_dir = "/Users/jaydeepdey/Desktop/capstone code/CNN_Denoising/MIRNet-Keras/dataset_polarimetric/train/PARAMS_POLAR/"
output_dir = "/Users/jaydeepdey/Desktop/capstone code/CNN_Denoising/MIRNet-Keras/dataset_polarimetric_annotated/"

annotate_images(csv_path, image_dir, output_dir)