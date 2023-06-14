# This script is in charge of training YOLOv8 model.
# Eventually it will also handle validation and hyperparameter searching.
from ultralytics import YOLO

if __name__ == "__main__":
    # Load a pretrained model
    model = YOLO('yolov8n.pt') # "n" for nano sized

    # Train the model with the SKU500 dataset
    model.train(data="SKU500.yaml", epochs=1)


    # Predict a single validation image
    #validatio_imgs = "SKU500/images/val"
    #results = model.predict(source=img, save=True)  # save plotted images

    model.val()  # It'll automatically evaluate the data you trained.
