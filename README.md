# SKU110K YOLOv8 Test #
In this repositoy I attempt to apply a YOLOv8 model on the SKU110K dataset. The purpose of this is to demonstrate my ability to proficiency in applying deep learning algorithms to real-world computer vision problems.

# Requirements #
Install YOLOv8 via the ultralytics pip package: https://github.com/ultralytics/ultralytics  
`pip install ultralytics`  
  
Download the SKU110K dataset and place it in the root directory: https://github.com/eg4000/SKU110K_CVPR19  
  
The root directory should look like this:
```
YOLOv8-SKU110K-Test  
    ├── README.md  
    ├── Prepare_SKU110K.py  
    ├── SKU110K_fixed/  
    ├── ...
```

# Prepare_SKU110K.py #
This script loads the SKU110k dataset and prepares it for input to YOLOv8. 500 images each for training/validation/testing are selected at random. The prepared dataset is put in a folder called SKU500.
<br> Usage: `python ./Prepare_SKU110K.py`
