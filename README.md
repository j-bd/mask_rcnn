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
