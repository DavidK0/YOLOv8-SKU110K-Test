# This script trains a YOLOv8 model on the SKU500 data set with hardcoded hyperparameters.
# This script takes no arguments.
# The hardcoded hyperparameters are the best values I found during
#   my shallow hyperparamer search.
# The final model is saved in "/runs" and can be tested with Test_YOLOv8.py

from ultralytics import YOLO

# I train the model with more epochs for deeper training than
#   was done during hyperparameter tuning.
num_epochs = 130

if __name__ == "__main__":
    # Load a pretrained model
    model = YOLO("yolov8s.pt") # "s" for small sized

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