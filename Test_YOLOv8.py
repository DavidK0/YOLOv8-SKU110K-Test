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

    print(dir(metrics))
    print(dir(metrics.box))
    print()

    print(metrics.box.all_ap , " all_ap")
    print(metrics.box.ap , " ap")
    print(metrics.box.ap50 , " ap50")
    print(metrics.box.ap_class_index , " ap_class_index")
    print(metrics.box.class_result , " class_result")
    print(metrics.box.f1 , " f1")
    print(metrics.box.fitness , " fitness")
    print(metrics.box.map , " map")
    print(metrics.box.map50 , " map50")
    print(metrics.box.map75 , " map75")
    print(metrics.box.maps , " maps")
    print(metrics.box.mean_results , " mean_results")
    print(metrics.box.mp , " mp")
    print(metrics.box.mr , " mr")
    print(metrics.box.nc , " nc")
    print(metrics.box.p , " p")
    print(metrics.box.r , " r")
    print(metrics.box.update , " update")