# This script continues to trains the given YOLOv8
#   on the SKU500 data set with hardcoded hyperparameters.
# It takes one argument, the weights file of a YOLOv8 model.
# The hardcoded hyperparameters are the best values I found during
#   a shallow hyperparamer search.
# The final model is saved in "/runs" and can be tested with Test_YOLOv8.py

from ultralytics import YOLO
import os
import sys

# Check for usage (needs a valid .pt file)
if len(sys.argv) < 2:
    print("Please provide a valid .pt file.")
    sys.exit(1)
filename = sys.argv[1]
if not os.path.isfile(filename) or not filename.lower().endswith(".pt"):
    print("{filename} is not a valid .pt file.")
    sys.exit(1)

# I train the model with more epochs for deeper training than
#   was done during hyperparameter tuning.
num_epochs = 100

if __name__ == "__main__":
    # Load a pretrained model
    model = YOLO(filename) # "n" for nano sized

    # Train the model with hardcoded hyperparameters
    model.train(data="SKU500.yaml", epochs=num_epochs,
        # Hyperparameters
        lr0=.098,
        lrf=.63,
        momentum=.67,
        weight_decay=.00073,
        warmup_epochs=0,
        warmup_momentum=.047,
        box=.056,
        cls=.57
    )