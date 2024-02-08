import os
import shutil

def create_dataset(train_folder, validation_folder, dataset_folder):
    # Create the dataset folder if it doesn't exist
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Iterate through the images in the train folder
    for filename in os.listdir(train_folder):
        train_image_path = os.path.join(train_folder, filename)
        validation_image_path = os.path.join(validation_folder, filename)

        # Extract the image name without the extension
        image_name = os.path.splitext(filename)[0]

        # Create a subdirectory in the dataset folder with the same name as the image
        dataset_subfolder = os.path.join(dataset_folder, image_name)
        os.makedirs(dataset_subfolder, exist_ok=True)

        # Copy and rename the images to the dataset subdirectory
        shutil.copy(train_image_path, os.path.join(dataset_subfolder, f"{image_name}_test.png"))
        shutil.copy(validation_image_path, os.path.join(dataset_subfolder, f"{image_name}_noisy.png"))


if __name__ == "__main__":
    train_folder = "./dataset_polarimetric/train/PARAM_POLAR"
    validation_folder = "./dataset_polarimetric_output/train/PARAM_POLAR"
    dataset_folder = "./dataset"

    create_dataset(train_folder, validation_folder, dataset_folder)