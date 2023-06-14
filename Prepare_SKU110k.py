# This script loads the SKU110k dataset and prepares it for input to YOLOv8
# 500 images each for training/validation/testing are selected at random
# The prepared dataset is put in a folder called SKU500

import os
import csv
import random
import shutil

# The location of the three data splits
SKU110K_train = r"SKU110K_fixed\annotations\annotations_train.csv"
SKU110K_val = r"SKU110K_fixed\annotations\annotations_val.csv"
SKU110K_test = r"SKU110K_fixed\annotations\annotations_test.csv"
SKU110K_images = r"SKU110K_fixed\images"

images_per_split = 500

# This method takes a CSV containing annotations in the SKU110K format
#   and outputs 500 at random to the output folder
def prepare_data_split(input_path, split_name):
    # Load the CSV data:
    with open(input_path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    # Set up the SKU500 directory
    subdirectory = "SKU500"
    os.makedirs(subdirectory, exist_ok=True)
    split_path = os.path.join(subdirectory,split_name)
    os.makedirs(split_path, exist_ok=True)
    images_output = os.path.join(split_path,"images")
    annotations_output = os.path.join(split_path,"annotations")
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(annotations_output, exist_ok=True)

    # Get unique image_names
    unique_image_names = list(set(row[0] for row in data))

    # Randomly select 500 image_names
    selected_image_names = random.sample(unique_image_names, images_per_split)

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
        annotations_output_path = os.path.join(split_path, "annotations", f"{image_name}.txt")

        # Check that the image exists in the SKU actually exists and copy it
        image_input_path = os.path.join(SKU110K_images, f"{image_name}.jpg")
        image_output_path = os.path.join(images_output, f"{image_name}.jpg")
        if not os.path.exists(image_input_path):
            print(f"The file {image_input_path} does not exist!", UserWarning)
            continue
        else:
            shutil.copy(image_input_path,image_output_path)

        with open(annotations_output_path, "w") as file:
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
                file.write(f"0,{center_x},{center_y},{width},{height}\n")

    print(f"{split_path} data created.")

if __name__ == "__main__":
    # Prepare the train, dev, and test splits
    prepare_data_split(SKU110K_train,"train")
    prepare_data_split(SKU110K_val,"val")
    prepare_data_split(SKU110K_test,"test")
