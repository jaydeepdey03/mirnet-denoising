import os
import pandas as pd
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def count_class_images(labels_csv, target_class):
    df_labels = pd.read_csv(labels_csv)
    class_images = df_labels[df_labels['class'] == target_class]
    return len(class_images)

def augment_images(input_dir, output_dir, labels_csv, target_classes):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Count the number of images for each class
    car_count = count_class_images(labels_csv, 'car')
    person_count = count_class_images(labels_csv, 'person')
    bike_count = count_class_images(labels_csv, 'bike')

    # Determine the target count for 'person' and 'bike' based on 'car' count
    target_person_count = min(car_count, person_count)
    target_bike_count = min(car_count, bike_count)

    # Create ImageDataGenerator with desired augmentation settings
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Augment 'person' images
    augment_class_images('person', target_person_count, input_dir, output_dir, labels_csv, datagen)

    # Augment 'bike' images
    augment_class_images('bike', target_bike_count, input_dir, output_dir, labels_csv, datagen)

def augment_class_images(target_class, target_count, input_dir, output_dir, labels_csv, datagen):
    df_labels = pd.read_csv(labels_csv)
    class_images = df_labels[df_labels['class'] == target_class]

    # Create output directory for the target class
    target_class_output_dir = os.path.join(output_dir, target_class)
    if not os.path.exists(target_class_output_dir):
        os.makedirs(target_class_output_dir)

    # Iterate through each image and perform augmentation
    for _, row in class_images.iterrows():
        if target_count <= 0:
            break

        image_path = os.path.join(input_dir, row['filename'])
        img = Image.open(image_path)

        # Expand dimensions to match expected input for datagen
        img_array = img.resize((row['width'], row['height']))
        img_array = img_array.convert('RGB')
        img_array = img_array.resize((row['width'], row['height']))
        img_array = img_array.convert('RGB')
        img_array = img_array.resize((row['width'], row['height']))
        img_array = img_array.convert('RGB')

        img_array = np.array(img_array)

        # Reshape to (1, height, width, channels)
        img_array = img_array.reshape((1,) + img_array.shape + (3,))

        # Generate augmented images and save to output directory
        for i, augmented_img_array in enumerate(datagen.flow(img_array, batch_size=1, save_to_dir=target_class_output_dir, save_prefix='aug_', save_format='png')):
            if i >= target_count:
                break

            # Get the augmented image and save the information in labels.csv
            augmented_img = Image.fromarray(augmented_img_array[0].astype('uint8'))
            new_filename = f"{row['filename'][:-4]}_aug_{i}.png"
            new_file_path = os.path.join(target_class_output_dir, new_filename)
            augmented_img.save(new_file_path)

            # Update labels.csv with the information of the new augmented image
            new_row = row.copy()
            new_row['filename'] = new_filename
            new_row['width'] = augmented_img.width
            new_row['height'] = augmented_img.height
            df_labels = df_labels.append(new_row, ignore_index=True)

        target_count -= 1

    # Save the updated labels.csv
    df_labels.to_csv(labels_csv, index=False)

# Example usage
input_directory = 'CNN/MIRNet-Keras/dataset_polarimetric/train/PARAM_POLAR'
output_directory = 'CNN/MIRNet-Keras'
labels_csv_path = 'CNN/MIRNet-Keras/dataset_polarimetric/train/labels.csv'
target_classes = ['person', 'bike']

augment_images(input_directory, output_directory, labels_csv_path, target_classes)
