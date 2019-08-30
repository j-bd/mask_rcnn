MODIFICATION ON GOING August 2019

# Implementation of Mask RCNN

You will find here a test application of M-RCNN deep learning method to an images detection challenge.


## Data source

I used data made available by airbus on Kaggle. The challenge name is ["Airbus Ship Detection Challenge"](https://www.kaggle.com/c/airbus-ship-detection).

The second version of data is organised as :
* A train folder with about 192 550 images,
* A test folder with about 15 600 images,
* A csv file named "train_ship_segmentations_v2.csv" with the mask of all train images,
* A csv file named "sample_submission_v2.csv" to show the shape of the output expected.


## Overview of Mask-RCNN pipeline

![Image of mrcnn](https://cdn-images-1.medium.com/max/2600/1*M_ZhHp8OXzWxEsfWu2e5EA.png)

Mask-RCNN [paper](https://arxiv.org/pdf/1703.06870.pdf) were submitt on 20 Mar 2017


## Implementing Mask RCNN for our own dataset and process for the training:

The following steps are necessary:
* Step 1: Clone [Mask RCNN repository](https://github.com/matterport/Mask_RCNN)
* Step 2: As mentionned by authors, we need first to create a sub-class config
based on ```mrcnn/config.py```. It will be not automatically implemented here. Thus, we create an inherit class and customed it following our computer capacity. This file is ```ship_config.py```
* Step 3: We create the structure of the project. We need:
   * A master project folder (PROJECT_DIR),
   * A sub-folder 'train' with all training images,
   * A sub-folder 'val' with all validation images,
   * A sub-folder 'backup_log' that will gather training logs during proceding (BACKUP).
* Step 4: We need a pre-trained weights file. If we choose Coco or Imagenet, we can
download it with provided commands (thanks to thte mrcnn authors!). We need to mention it when launching the training. We can also used a last file (for instance, if we want to resume a training)
* Step 5: We organised images between "train" and "val" (validation) folders. We
offer the possibility to choose the amount of data while respecting the ratio
between images with and without mask
* Step 6: We charge the configuration file create in step 2
* Step 7: We instantiate the MaskRCNN Class present in mrcnn/model.py
* Step 8: We load the pre-trained weights file. See step 4
* Step 9: We instantiate the ShipDataset class (based on Dataset class in
mrcnn/utils.py) for:
   * managing training data
   * managing validation data
* Step 10: Training can be launch

Two parameters can be modulated for the training :
* Numbers of steps per epoch in ship_config.py
* Numbers of epochs in ship_functions.py
