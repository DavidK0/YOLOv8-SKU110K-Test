# v8-v5 Comparison #
In this section I perform a comparison between three YOLO models (v5 nano, v5 small, and v8 nano) with regard to their inference performance and speed. This comparison builds on my initial tests with YOLOv8 (using the same hyperparameter tuning values and SKU500 dataset).

# Requirements #
As well as [YOLOv8](https://github.com/ultralytics/ultralytics) and [SKU500](https://github.com/DavidK0/YOLOv8-SKU110K-Test/blob/main/README.md#dataset), install [YOLOv5](https://github.com/ultralytics/yolov5) and place it in /v8-v5_Comparison/:  
```
cd v8-v5_Comparison
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

# Methods #
I trained three YOLO models:
* YOLOv8 nano (from yolo5n.pt)
* YOLOv5 nano (from yolo5n.pt)
* YOLOv5 small (from yolo5s.pt)

I trained all three models for 130 epochs and a batch size of 16. To set the hyperparameters, I used the best hyperparameters that I previously found during hyperparameter tuning. I used these values for all three models. For hyperparameters that were not included in my hyperparameter search, I used the same default values for all three models, and got the default values from the [Ultralytics `default.config`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/cfg/default.yaml). For hyperparameters that are present in YOLOv5 but not YOLOv8, I used the default values from the [Ultralyrics `hyp.scratch-low.yaml`](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml).  
The training data was the training split of the SKU500 dataset, the subset of SKU110K that I used for initially taining YOLOv8. The best model from each train run was saved and used for the comparison. The mAP, precision, and recall for YOLOv5 are calculated by running `python .\v8-v5_Comparison\yolov5\val.py --data .\SKU500_v5.yaml --weights .\v8-v5_Comparison\Final_YOLOv5-nano.pt --task test`. For YOLOv8 the evaluation metrics are calculated by using `Test_YOLOv8.py`.  
To find the inference speed for the YOLOv8 model I used Ultralytic's built-in validation method, which returns the inference time. I validated the model on the 50 images from the validation split of SKU500, and repeated that 25 times to get the final average. To find the inference speed for the YOLOv5 models I


# Results #
|Model       |mAP50|Precision |Recall|Inference Speed\(ms\)|FPS|
|------------|-----|----------|------|---------------------|---|
|YOLOv8-nano |88.4%|86.0%     |75.4% |                     |    
|YOLOv5-nano |79.7%|86.0%     |69.5% |                     |
|YOLOv5-small|82.0%|86.2%     |73.8% |                     |

The PR curves for VOLOv8-nano, VOLOv5-nano, and YOLOv5-small respectively are given below.
<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/3310d2ab-124e-4b00-a8ac-db7906f61c19" alt=""YOLOv8-nano PR curve on SKU500 test split"" width="284" height="188">
<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/8f94cfa5-0d8b-4f02-9799-70d07a3389ae" alt="YOLOv5-nano PR curve on SKU500 test split" width="284" height="188">
0K-Test/assets/9288945/8f94cfa5-0d8b-4f02-9799-70d07a3389ae" alt="YOLOv5-nano PR curve on SKU500 test split" width="284" height="188">
<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/3eeb49a6-3ea6-4dc6-88a0-9c483111b0bd" alt=""YOLOv5-small  PR curve on SKU500 test split"" width="284" height="188">

# Discussion and Conclusion #
\<insert summary\>
