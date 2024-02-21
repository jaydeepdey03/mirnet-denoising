import os
import random
import shutil

def shuffle_data(src_folder, dest_folder, fraction):
    # Get list of all image files in source folder
    image_files = [f for f in os.listdir(os.path.join(src_folder, 'images')) if f.endswith(('.png'))]
    
    # Shuffle the files randomly
    random.shuffle(image_files)
    
    # Calculate the number of files to move
    num_files_to_move = int(len(image_files) * fraction)
    
    # Move files from source to destination folder
    for i in range(num_files_to_move):
        image_file = image_files[i]
        label_file = image_file.split('.')[0] + '.xml'
        
        src_image_path = os.path.join(src_folder, 'images', image_file)
        src_label_path = os.path.join(src_folder, 'labels_xml', label_file)
        
        dest_image_path = os.path.join(dest_folder, 'images', image_file)
        dest_label_path = os.path.join(dest_folder, 'labels_xml', label_file)
        
        # Move the files
        shutil.move(src_image_path, dest_image_path)
        shutil.move(src_label_path, dest_label_path)

if __name__ == "__main__":
    train_folder = "./CNN/MIRNet-Keras/dataset_polarimetric_modified/train"
    test_folder = "./CNN/MIRNet-Keras/dataset_polarimetric_modified/test"
    
    # Fraction of data to shuffle (e.g., 0.2 for 20%)
    shuffle_fraction = 0.2
    
    # Shuffle data from train to test
    shuffle_data(train_folder, test_folder, shuffle_fraction)
    
    # Shuffle data from test to train
    shuffle_data(test_folder, train_folder, shuffle_fraction)
