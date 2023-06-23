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
The script `v8_Speed_Test.py` is located in the repository root in order to make it work with YOLOv8.

# Methods #
I trained three YOLO models:
* YOLOv8 nano (from yolo5n.pt)
* YOLOv5 nano (from yolo5n.pt)
* YOLOv5 small (from yolo5s.pt)

I trained all three models for 130 epochs and a batch size of 16. To train the YOLOv5 models I use `/yolov5/train.py`. The final models can be found at `Final_YOLOv5-nano.pt` and `Final_YOLOv5-small.pt`. The YOLOv8 model is the same model that I previously train (found at `../Final_YOLOv8-nano.pt`).

To set the hyperparameters, I used the best hyperparameters that I previously found during hyperparameter tuning. I used these values for all three models. For hyperparameters that were not included in my hyperparameter search, I used the same default values for all three models, and got the default values from the [Ultralytics `default.config`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/cfg/default.yaml). For hyperparameters that are present in YOLOv5 but not YOLOv8, I used the default values from the [Ultralyrics `hyp.scratch-low.yaml`](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml).

The training data was the training split of the SKU500 dataset, the subset of SKU110K that I used for initially taining YOLOv8. All training and inference was done with CUDA on a NVIDIA GeForce RTX 3060. The best model from each training run was saved and used for the comparison. The mAP, precision, and recall for YOLOv5 are calculated using `/yolov5/val.py`. For YOLOv8 the evaluation metrics are calculated by using `Test_YOLOv8.py`.

To find the inference speed for the YOLOv8 model I used `v8_Speed_Test.py` which uses Ultralytic's built-in validation method, which returns the inference time. I validated the model on the 50 images from the test split of SKU500, and repeated that 25 times to get the final average. To find the inference speed for the YOLOv5 models I used `v5_Speed_Test.py`, which also has the models evaluate all 50 test images 25 times.

# Results #
Each of the models are listed in the table below, along with evaluation and speed metrics on the SKU500 test set.
|Model       |mAP50|Precision |Recall|Inference Speed|FPS  |
|------------|-----|----------|------|---------------|-----|
|YOLOv8-nano |88.4%|86.0%     |75.4% |8.73           |114.5|
|YOLOv5-nano |79.7%|86.0%     |69.5% |7.88 ms        |126.9|
|YOLOv5-small|82.0%|86.2%     |73.8% |11.09 ms       |90.2 |

The PR curves for VOLOv8-nano, VOLOv5-nano, and YOLOv5-small respectively are given below.

<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/3310d2ab-124e-4b00-a8ac-db7906f61c19" alt="YOLOv8-nano PR curve on SKU500 test split" width="284" height="188">

<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/8f94cfa5-0d8b-4f02-9799-70d07a3389ae" alt="YOLOv5-nano PR curve on SKU500 test split" width="284" height="188">

<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/3eeb49a6-3ea6-4dc6-88a0-9c483111b0bd" alt="YOLOv5-small  PR curve on SKU500 test split" width="284" height="188">

# Discussion and Conclusion #
The results above show that the latest model, YOLOv8, offers the highest performance in terms of mAP50 with a score of 88.4%, although it comes in second place in terms of speed with a inference speed of 8.7 ms. The two YOLOv5 models had lower mAP50 scores than YOLOv8 but faster inference speeds. The larger YOLOv5 model, YOLOv5-small, had a mAP50 of 82% and an inference speed of 11.1. The smaller model, YOLOv5-nano, had a mAP50 of 79.7% and an inference speed of 7.9 ms, making it the model with the lowest average precision but fastest inference speed.

In conclusion, my tests show that accuracy and speed is a trade-off within the YOLO family of models. YOLOv8 exhibits the hightest detection accuracy and decent inference speed, but not the fastest speed. YOLOv5 has comparatively fast inference speeds but worse accuracy. Within YOLOv5, larger models yeild higher accuracy but slower inference speeds.
