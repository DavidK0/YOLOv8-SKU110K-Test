# SKU110K YOLOv8 Test #
In this repositoy I attempt to apply a YOLOv8 model on the SKU110K dataset. The purpose of this is to demonstrate my ability to apply deep learning algorithms to real-world computer vision problems.

# Requirements #
Install [YOLOv8](https://github.com/ultralytics/ultralytics) via the ultralytics pip package:  
`pip install ultralytics`  

Download the [SKU110K dataset](https://github.com/eg4000/SKU110K_CVPR19 ) and place it in the root directory. The root directory should look like this:  
```
YOLOv8-SKU110K-Test  
    ├── README.md  
    ├── Prepare_SKU110K.py
    ├── Train_YOLOv8.py
    ├── SKU110K_fixed/
    ├── runs/
    ├── ...
```

# Dataset #
&nbsp;&nbsp;&nbsp;&nbsp;My oringinal intention was train a YOLOv8 model on the [Unitail-Det dataset](https://unitedretail.github.io/Unitail-Det/), however the object detection mode of YOLOv8 only supports axis-aligned bounding boxes and not the more flexible qraudrilaterals that Unitail-Det offers. Instead I used the parent dataset of Unitail-Det, [SKU110K](https://github.com/eg4000/SKU110K_CVPR19 ).  
&nbsp;&nbsp;&nbsp;&nbsp;To format the SKU110k dataset and prepare it for input to YOLOv8, use `python ./Preprocess_SKU110K.py`. The main preprocessing step is to reformat bounding boxes locations from (corner 1, corner 2) to (center, size). I reuse the train/validate/test split from the oringal dataset but sample randomly from those splits to get a smaller dataset. 400 images are taken from the training split, 50 from validation, and 50 from testing for a total of 500 images. The prepared dataset is put in "src/datasets/SKU500" and its .yaml file is put in the repository root.  
&nbsp;&nbsp;&nbsp;&nbsp;A couple of the original images in the SKU110K dataset are corrupt and unable to be read by YOLOv8. These corrupt images are not handled by the pre-processor but instead are skipped during training. Corrupt images are rare and have a small effect on training.

# Performance Evaluation #
A common evaluation metrics for computer is the mAP, which is the mean AP over all classes. The dataset I used has only one class ('object'), which means the mAP is equivalent to just the AP. During hyperparameter tuning, I evaluate the model using the mAP metric with an IoU threshold of 50%. For testing and reporting the models final performance I use a few IoU thresholds.

# Hyperparameter tuning #
`python ./Train_YOLOv8.py`
My first attempt at hyperparameter tuning was to use [Ray Tune](https://docs.ray.io/en/latest/tune/index.html), a hyperparameter tuning library with integration with YOLOv8. Unfortunately, I was unable to overcome a [techincal issue related to job creation](https://github.com/ray-project/ray/issues/21994).  
Instead, I performed my own hyperparameter tuning. I randomly searched 8 hyperparameter values and ran 50 trials, each up to 20 epochs long. 50 trials at 20 epochs each is not a very in-depth hyperparameter search, but it is enough to see the effects of the hyperparameter. Using CUDA on an Nvidia GeForce RTX 3060 hyperparameter tuning took 7.5 hours. The hyperparameters that I searched, their search ranges, and the best found values are listed below.  
|Parameter            |Range      |Best|
|---------------------|-----------|----|
|Initial learning rate|.00001 - .1|.001|
|Final learning rate  |0.01 - 1   |.01|
|Momentum             |0.6 - 0.98 |.8|
|Weight decay         |0 - 0.001  |.001|
|Warmup epochs        |0 - 5      |.2|
|Warmup momentum      |0 - 0.95   |.5|
|Box loss weight      |0.02 - 0.2 |.05|
|Class loss weight    |0.2 - 4    |1|


# Results #
The finalized model is the best model found during hyperparameter tuning. I evaluated the finalized model on the test set. The AP at different IoU thresholds is given below.
|mAP50|.7|
|mAP75|.6|
|mAP50-95|.65|

# Conclusion #
In conculusion, I have shown that a YOLOv8 model can be trained to produce image detections for densely packed objects on the SKU110K dataset, demonstrating my ability to tackle real-world computer vision problems using deep learning algorithms. The SKU110K dataset was preprocessed and reformatted to suit the YOLOv8's input requirements. A few corrupt images were encountered in the dataset but they had minimal impact on training. Hyperparameter tuning was performed the best model was saved. The best model achieved a final mAP50 score of __ and a mAP50-95 of __
