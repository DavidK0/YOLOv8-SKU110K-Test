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
    ├── Train_YOLOv8.py
    ├── SKU110K_fixed/
    ├── runs/
    ├── ...
```

# Prepare_SKU110K.py #
This script loads the SKU110k dataset and prepares it for input to YOLOv8. 500 images each for training/validation/testing are selected at random. The prepared dataset is put in a folder called SKU500.
<br> Usage: `python ./Prepare_SKU110K.py`

# Train_YOLOv8.py #
After pre-processing the SKU110K data, run `python ./Train_YOLOv8.py` to test basic training and validation on YOLOv8.

# Validation #
Here is where I will explain my validation strategy.

# Hyperparameter tuning #
My first attempt at hyperparameter tuning was to use [Ray Tune](https://docs.ray.io/en/latest/tune/index.html), a hyperparameter tuning library with integration with YOLOv8. Unfortunately, I was unable to overcome a [techincal issue related to job creation](https://github.com/ray-project/ray/issues/21994).  
Instead, I performed my own hyperparameter tuning. I randomly searched 5(?) hyperparameter values and ran 50(?) trials. The hyperparameters, ranges, and best values are listed below.  
| Parameter      | Range | Best |
| ----------- | ----------- | - |
| l0      | .0001-.01       | .001 |

# Results #
Here is where run the finalized model on the test set and report the results.
