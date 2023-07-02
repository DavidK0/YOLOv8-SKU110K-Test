# v8-v5 Comparison #
In this section I perform a comparison between four YOLO models (v5 nano, v5 small, v8 nano, and v8 small) with regard to their inference performance and speed. This comparison builds on my initial tests with YOLOv8 (using the same hyperparameter tuning values and SKU500 dataset).

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
I trained four YOLO models:
* YOLOv8 nano (from yolo5n.pt)
* YOLOv8 small (from yolo5s.pt)
* YOLOv5 nano (from yolo5n.pt)
* YOLOv5 small (from yolo5s.pt)

I trained all four models for 130 epochs and a batch size of 16. To train the YOLOv5 models I use `/yolov5/train.py`. The final models can be found at `Final_YOLOv5-nano.pt` and `Final_YOLOv5-small.pt`. The YOLOv8-nano model is the same model that I previously train (found at `../Final_YOLOv8-nano.pt`). The YOLOv8-small model was trained with `Train_YOLOv8-small.py`, and is found at `../Final_YOLOv8-small.pt`.

To set the hyperparameters, I used the best hyperparameters that I previously found during hyperparameter tuning. I used these values for all four models. For hyperparameters that were not included in my hyperparameter search, I used the same default values for all four models, and got the default values from the [Ultralytics `default.config`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/cfg/default.yaml). For hyperparameters that are present in YOLOv5 but not YOLOv8, I used the default values from the [Ultralyrics `hyp.scratch-low.yaml`](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml).

The training data was the training split of the SKU500 dataset, the subset of SKU110K that I used for initially taining YOLOv8. All training and inference was done with CUDA on a NVIDIA GeForce RTX 3060. The best model from each training run was saved and used for the comparison. The mAP, precision, and recall for YOLOv5 are calculated using `/yolov5/val.py`. For YOLOv8 the evaluation metrics are calculated by using `Test_YOLOv8.py`.

To find the inference speed for the YOLOv8 models I used `v8_Speed_Test.py` which uses Ultralytic's built-in validation method, which returns the inference time. I validated the model on the 50 images from the test split of SKU500, and repeated that 25 times to get the final average. To find the inference speed for the YOLOv5 models I used `v5_Speed_Test.py`, which also has the models evaluate all 50 test images 25 times.

# Results #
Each of the models are listed in the table below, along with evaluation and speed metrics on the SKU500 test set.
|Model       |mAP50|Precision |Recall|Inference Speed|FPS  |
|------------|-----|----------|------|---------------|-----|
|YOLOv8-nano |88.4%|86.0%     |75.4% |8.73           |114.5|
|YOLOv8-small|88.0%|89.2%     |80.3% |11.6           |86.2 |
|YOLOv5-nano |79.7%|86.0%     |69.5% |7.88 ms        |126.9|
|YOLOv5-small|82.0%|86.2%     |73.8% |11.09 ms       |90.2 |

The PR curves for VOLOv8-nano, YOLOv8-small, VOLOv5-nano, and YOLOv5-small respectively are given below.

<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/3310d2ab-124e-4b00-a8ac-db7906f61c19" alt="YOLOv8-nano PR curve on SKU500 test split" width="284" height="188">

<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/df6dbfc2-3b34-45b2-89d7-d3e579179d40" alt="YOLOv5-nano PR curve on SKU500 test split" width="284" height="188">

<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/8f94cfa5-0d8b-4f02-9799-70d07a3389ae" alt="YOLOv5-nano PR curve on SKU500 test split" width="284" height="188">

<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/3eeb49a6-3ea6-4dc6-88a0-9c483111b0bd" alt="YOLOv5-small  PR curve on SKU500 test split" width="284" height="188">

# Discussion #
The results above show that the latest model, YOLOv8, offers the highest performance in terms of mAP50 at the cost of speed. YOLOv8-nano had the highest with a score of 88.4%, although it comes in second place in terms of speed with a inference speed of 8.7 ms. YOLOv8-small did not outperform YOLOv8-nono, possibly due to the small size of the training data. YOLOv8-small had an mAP of 88.0% and the slowest inference time at 11.6 ms. The two YOLOv5 models had lower mAP50 scores than YOLOv8 but faster inference speeds. The larger YOLOv5 model, YOLOv5-small, had a mAP50 of 82% and an inference speed of 11.1. The smaller model, YOLOv5-nano, had a mAP50 of 79.7% and an inference speed of 7.9 ms, making it the model with the lowest average precision but fastest inference speed.

I examined a few of the output labels assigned by the models during testing. The first thing I noticed is that the models tended to add extra bounding boxes, especially in the background. Sometimes this explainable by oddities in the data, for example in the image below we can see that in the true annotation, bounding boxes cuts off at an arbitrary depth from the camera. The models however continue to detect objects even as the shelves approach the background. Furthermore the larger models \(small\) produce more extra object detections than the smaller ones \(nano)\.

<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/719dccbe-233c-4b0a-8296-b6942f3da73f" alt="Visualization 1" width="725" height="384">


Another challenge for the models was objects that do not have a face squarly aligned with the shelves and clearly visible. In the image below there is a stack of cloths, but they are crumbled and slightly obscured. All the models struggle to identify them. In this case, the two large models do a worse job at identify the objects than the smaller models. Another trend that can be seen in the image below is that both sizes of YOLOv5 were more likely to produce overlapping bounding boxes.
 
<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/6afd7270-59eb-42a6-aaa9-a334ee885713" alt="Visualization 2" width="732" height="270">


In the image below are four example of annotations that were challenging for all the models. First, inconsitantly place annotations are annotations that cover only one item in a group of items, or that cover an item at the back of the shelf that normally would not be annotated. These are likely challenging because the model can not learn whether or not such items should be included. Secondly, items which are not products \(like signs\) are sometimes found on shelves which are often incorrectly identified as objects. Third, the annotations sometimes skip over some items or even entire shelves. In this case, all four models picked up on all the items on that shelf. Fourth, the annotations do not always cover the entire items, in which case the models produce larger bounding boxes than desired.

<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/448c6de2-ba91-4881-b2c1-afa23732d207" alt="Visualization 3" width="432" height="358">

# Conclusion #
In conclusion, my tests show that accuracy and speed is a trade-off within the YOLO family of models. YOLOv8 exhibits the hightest detection accuracy and decent inference speed, but not the fastest speed. YOLOv5 has comparatively fast inference speeds but worse accuracy. Within one version of YOLO, larger models yeild higher accuracy but slower inference speeds. YOLOv8-small has a roughly 6 percentage point increase over YOLOv5-small while only dropping its inference time by about 0.5 ms, so although YOLOv8-small is slower, the cost is likely worth the performance gain. As for errors, YOLOv5 was more likely to produce overlapping bounding boxes, and larger models were more likely to detect objects in the background. Some other errors were common to both models and can be attributed to oddities in the dataset \(ex. depth-of-field cuttof, missing labels\).

