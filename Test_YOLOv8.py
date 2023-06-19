# This script loads the given YOLOv8 model and evaluates it on the SKU500 test set.
# This script takes one argument: the weights file of the model to evaluate (ex. "best.pt")

from ultralytics import YOLO
import sys
import os

# Check for usage (needs a valid .pt file)
if len(sys.argv) < 2:
    print("Please provide a valid .pt file.")
    sys.exit(1)
filename = sys.argv[1]
if not os.path.isfile(filename) or not filename.lower().endswith(".pt"):
    print("{filename} is not a valid .pt file.")
    sys.exit(1)

if __name__ == "__main__":
    # Load the given model
    model = YOLO(filename)

    # Evaluate the model on the SKU500 test split
    # A new validation run will be put in "runs"
    metrics = model.val(split="test")

    # Print some key metrics
    print(f"All AP: {metrics.box.all_ap}")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")
    print(f"Mean results: {metrics.box.mean_results}")
    print(f"Mean Precsion: {metrics.box.mp}")
    print(f"Mean recall: {metrics.box.mr}")