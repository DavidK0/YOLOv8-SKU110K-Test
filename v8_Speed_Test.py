# This script calculates a YOLOv8's average inference time on the SKU500 dataset.
# This script takes two arguments:
#   A valid .pt file from which to load the YOLOv8 model
#   The split on which to evaluate ("train", "val", or "test")

from ultralytics import YOLO
import sys
import os

num_val_runs = 25 # The number of validation runs to take an average over

# Check for usage (needs a valid .pt file)
if len(sys.argv) < 3:
    print("Please provide a valid .pt file and split name.")
    sys.exit(1)
filename = sys.argv[1]
if not os.path.isfile(filename) or not filename.lower().endswith(".pt"):
    print("{filename} is not a valid .pt file.")
    sys.exit(1)
split_name = sys.argv[2]
if not split_name in ["train", "val", "test"]:
    print("Please provide a valid split name (\"train\", \"val\", or \"test\")")
    sys.exit(1)

if __name__ == "__main__":
    # Load the given model
    model = YOLO(filename)

    total_time = 0
    for run in range(num_val_runs):
        # Display progress
        print(f"Progress: {run + 1}/{num_val_runs} ")

        # Evaluate the model on the given split of the SKU500 dataset
        metrics = model.val(split=split_name)

        # Get the inference time
        total_time += metrics.speed["inference"]

    # Find the average inference time
    average_time = total_time / num_val_runs
    print(f"Average inference time: {average_time:.2f}")
    print(f"Average FPS: {1000 / average_time:.2f}")