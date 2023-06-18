from ultralytics import YOLO

# Load the previous best
model = YOLO('yolov8n.pt') # "n" for nano sized
model.train(data="SKU500.yaml", epochs=100)
metrics = model.val()
mAP50 = metrics.box.map50