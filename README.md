# SKU110K YOLOv8 Test #
In this repositoy I attempt to apply a YOLOv8 model to the SKU110K dataset. The purpose of this is to demonstrate my ability to apply deep learning algorithms to real-world computer vision problems.

# Requirements #
Install [YOLOv8](https://github.com/ultralytics/ultralytics) via the ultralytics pip package:  
`pip install ultralytics`  

Download the [SKU110K dataset](https://github.com/eg4000/SKU110K_CVPR19 ) and place it in the root directory. The root directory should look like this:  
```
YOLOv8-SKU110K-Test  
    ├── SKU110K_fixed/
    ├── Preprocess_SKU110K.py
    ├── Train_YOLOv8.py
    ├── Finalize_YOLOv8.py
    ├── Test_YOLOv8.py
    ├── README.md  
    ├── ...
```

# Dataset #
&nbsp;&nbsp;&nbsp;&nbsp;My oringinal intention was train a YOLOv8 model on the [Unitail-Det dataset](https://unitedretail.github.io/Unitail-Det/), however the object detection mode of YOLOv8 only supports axis-aligned bounding boxes and not the more flexible qraudrilaterals that Unitail-Det offers. Instead I used the parent dataset of Unitail-Det, [SKU110K](https://github.com/eg4000/SKU110K_CVPR19 ). The images in the SKU110K dataset are a superset of those in the Unitail-Det dataset but are still in the same domain of densley packed objects on shelves.  
&nbsp;&nbsp;&nbsp;&nbsp;To format the SKU110k dataset and prepare it for input to YOLOv8, I used `Preprocess_SKU110K.py`. The main preprocessing step is to reformat bounding boxes locations from (corner 1, corner 2) to (center, size). I reuse the train/validate/test split from the oringal dataset but sample randomly from those splits to get a smaller dataset. 400 images are taken from the training split, 50 from validation, and 50 from testing for a total of 500 images. The prepared dataset is put in "src/datasets/SKU500" and its .yaml file is put in the repository root.  
&nbsp;&nbsp;&nbsp;&nbsp;A couple of the original images in the SKU110K dataset are corrupt and unable to be read by YOLOv8. These corrupt images are not handled by the pre-processor but instead are skipped during training. Corrupt images are rare and have a small effect on training.

# Performance Evaluation #
A standard evaluation metrics in computer vision problems is the mAP, which is the mean AP over all classes. The dataset I used has only one class ('object'), which means the mAP is equivalent to just the AP. During hyperparameter tuning, I evaluate the model using the mAP metric with an IoU threshold of 50%. For testing and reporting the models final performance I use a few IoU thresholds.

# Hyperparameter tuning #
My first attempt at hyperparameter tuning was to use [Ray Tune](https://docs.ray.io/en/latest/tune/index.html), a hyperparameter tuning library with integration with YOLOv8. Unfortunately, I was unable to overcome a [techincal issue related to job creation](https://github.com/ray-project/ray/issues/21994).  
Instead, I performed my own hyperparameter tuning by using `Train_YOLOv8.py`. Starting with a pre-trained nano-sized YOLOv8 model, I randomly searched 8 hyperparameter values and ran 35 trials, each up to 15 epochs long. 35 trials at 15 epochs each is not a very in-depth hyperparameter search, but it is enough to see the effects of the hyperparameters. Using CUDA on an Nvidia GeForce RTX 3060 hyperparameter tuning took 4.1 hours. The hyperparameters that I searched, their search ranges, and the best found values are listed below.  
|Parameter            |Range      |Best   |
|---------------------|-----------|-------|
|Initial learning rate|.00001 - .1|0.098  |
|Final learning rate  |0.01 - 1   |0.63   |
|Momentum             |0.6 - 0.98 |0.67   |
|Weight decay         |0 - 0.001  |0.00073|
|Warmup epochs        |0 - 5      |0      |
|Warmup momentum      |0 - 0.95   |0.047  |
|Box loss weight      |0.02 - 0.2 |0.056  |
|Class loss weight    |0.2 - 4    |0.57   |

# Testing and Results #
To finalize the model, I continued to train the best model from hyperparameter tuning for and additional 100 epochs by using `Finalize_YOLOv8.py`. The weights file of the final model can be found at `Final-yolov5n.pt`. I evaluated the finalized model on the test set by using `Test_YOLOv8.py`. The AP (equivalent to mAP for one class) at different IoU thresholds is given below as well as the mean precision and mean recall.  
|Metric|Score|
|-------|-----|
|AP50   |0.884|
|AP75   |0.547|
|AP50-95|0.503|
|MP     |0.860|
|MR     |0.754|

The PR curve is given below.  
<img src="https://github.com/DavidK0/YOLOv8-SKU110K-Test/assets/9288945/3310d2ab-124e-4b00-a8ac-db7906f61c19" alt="Alt Text" width="563" height="376">

# Conclusion #
In conculusion, I have shown that a YOLOv8 model can be trained to produce image detections for densely packed objects on the SKU110K dataset, demonstrating my ability to tackle real-world computer vision problems using deep learning algorithms. The SKU110K dataset was preprocessed and reformatted to suit the YOLOv8's input requirements. A few corrupt images were encountered in the dataset but they had minimal impact on training. Hyperparameter tuning was performed and the best model was selected and further trained. The finalized model achieved a mAP50 score of 88.4% on the testing data.
