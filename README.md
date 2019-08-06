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


## Global organisation of the repository

The folder **mrcnn** gather the heart of the Mask RCNN functioning. It comes from the great work of [Mask R-CNN library](https://github.com/matterport/Mask_RCNN) that Matterport built.

The files ```learning_ship.py``` and ```ship_detection.py``` are mostly from the work of [gabrielgarza](https://github.com/gabrielgarza/Mask_RCNN/tree/master/samples/ship)

Finally, the file ```data_preprocessing.py``` is the one I did in order to select few images in each Airbus *test* and *train* folders. Indeed, as I worked on a weak computer, I could not allow to train the model on the full data. Anyway, the aim was to understand the full process.


## Overview of Mask-RCNN pipeline

![Image of mrcnn](https://cdn-images-1.medium.com/max/2600/1*M_ZhHp8OXzWxEsfWu2e5EA.png)
