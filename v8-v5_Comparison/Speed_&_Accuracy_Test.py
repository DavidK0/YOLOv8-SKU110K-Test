# This runs a speed and performance comparison between two versions of YOLO

# for YOLOv8
from ultralytics import YOLO

# for YOLOv5
import torch

# Model

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.


# I train the model with more epochs for deeper training than
#   was done during hyperparameter tuning.
num_epochs = 100

if __name__ == "__main__":
    # Load a pretrained v8 model
    model = YOLO("yolov5n.pt") # "n" for nano sized
    # Load a pretrained v5-n model
    model = torch.hub.load("ultralytics/yolov5", "yolov5n")
    # Load a pretrained v5-s model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Train with default hyperparameters
    model.train(data="SKU500.yamls", epochs=num_epochs,
    )