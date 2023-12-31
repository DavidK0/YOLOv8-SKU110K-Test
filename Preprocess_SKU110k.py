# This script loads the SKU110k dataset and prepares it for input to YOLOv8
# 500 images each for training/validation/testing are selected at random
# The prepared dataset is put in a folder called SKU500
# This script takes no arguments.

import os
import csv
import random
import shutil

# The location of the SKU110K dataset
SKU110K = r"SKU110K_fixed"
SKU110K_train = os.path.join(SKU110K, r"annotations\annotations_train.csv")
SKU110K_val = os.path.join(SKU110K, r"annotations\annotations_val.csv")
SKU110K_test = os.path.join(SKU110K, r"annotations\annotations_test.csv")
SKU110K_images = os.path.join(SKU110K, r"images")

# Output location
output_directory = "datasets/SKU500"

# The seed for randomizing the splits (None defaults to system time)
random_seed = None
random.seed(random_seed)

# This method takes a CSV containing annotations in the SKU110K format
#   and outputs 500 at random to the output folder
def prepare_data_split(input_path, split_name, images_in_split):
    # Load the CSV data:
    with open(input_path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    # Set up the output directory
    os.makedirs(output_directory, exist_ok=True)
    images_path = os.path.join(output_directory,"images")
    labels_path = os.path.join(output_directory,"labels")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    images_output = os.path.join(images_path, split_name)
    labels_output = os.path.join(labels_path,split_name)
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)

    # Get unique image_names
    unique_image_names = list(set(row[0] for row in data))

    # Randomly select 500 image_names
    selected_image_names = random.sample(unique_image_names, images_in_split)

    # Filter the data for selected image_names
    selected_data = [row for row in data if row[0] in selected_image_names]

    # Output the selected data to .txt files
    grouped_data = {}
    for row in selected_data:
        image_name = row[0]
        if image_name not in grouped_data:
            grouped_data[image_name] = []
        grouped_data[image_name].append(row)

    # Output each grouped data into a separate .txt file
    for image_name, data in grouped_data.items():
        # Remove the file extension from the name
        image_name = os.path.splitext(image_name)[0]

        # Create the text file for the annotations
        labels_output_path = os.path.join(labels_output, f"{image_name}.txt")

        # Check that the image exists in the SKU actually exists and copy it
        image_input_path = os.path.join(SKU110K_images, f"{image_name}.jpg")
        image_output_path = os.path.join(images_output, f"{image_name}.jpg")
        if not os.path.exists(image_input_path):
            print(f"The file {image_input_path} does not exist!", UserWarning)
            continue
        else:
            shutil.copy(image_input_path,image_output_path)

        with open(labels_output_path, "w") as file:
            # Iterate over each object in the image
            for row in data:
                # SKU-110k annotations are stored with the following format:
                #   image_name, x1, y1, x2, y2, class, image_width, image_height
                # We need them to be:
                #   0, center_x, center_y, width, height
                #   and also normalized by the size of the image

                # Find the center of the bounding box
                center_x = (float(row[1]) + float(row[3])) / 2
                center_y = (float(row[2]) + float(row[4])) / 2

                # Find the width and height of the bounding box
                width = float(row[3]) - float(row[1])
                height = float(row[4]) - float(row[2])

                # Normalize everything by the height of the image
                center_x /= float(row[6])
                center_y /= float(row[7])
                width /= float(row[6])
                height /= float(row[7])

                # Write the row to the file
                file.write(f"0 {center_x} {center_y} {width} {height}\n")

    print(f"{split_name} data created.")

if __name__ == "__main__":
    # Prepare the train, dev, and test splits
    prepare_data_split(SKU110K_train,"train", 400)
    prepare_data_split(SKU110K_val,"val", 50)
    prepare_data_split(SKU110K_test,"test", 50)

    # Produce the YAML file
    yaml_path = os.path.join("./", "SKU500.yaml")
    with open(yaml_path, "w") as file:
        file.write("path: SKU500")
        file.write("\ntrain: images/train")
        file.write("\nval: images/val")
        file.write("\ntest: images/test")
        file.write("\n")
        file.write("\nnames:")
        file.write("\n  0: object")
    
    # For unknown reasons, Ultralytics does not work if there is not
    #   a folder called "runs" in the repository root
    os.makedirs("runs", exist_ok=True)