#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:56:45 2019

@author: j-bd

The aim is to create a lighter dataset to run on laptop.
After downloading the kaggle data (https://www.kaggle.com/c/airbus-ship-detection),
from nearly 200 000 images, we will pick up around 200 images
"""
import shutil

import pandas as pd

# =============================================================================
# Loading target data from training directory
# =============================================================================

# Root directory of the project
IMAGE_DIR = "/path/to/airbus_ship/images/directory"
PROJECT_DIR = "path/to/the/new/directory"
FILE_NAME = "train_ship_segmentations.csv" # provide in airbus dataset
FILE_TEST_NAME = "sample_submission.csv" # provide in airbus dataset

dataset = pd.read_csv(IMAGE_DIR + "/" + FILE_NAME)
test = pd.read_csv(IMAGE_DIR + "/" + FILE_TEST_NAME)

#Keep only images with ship
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)


# =============================================================================
# Choosing the right amount of data in accordance with the computer power used.
# Save the choice under three folders : 'train', 'val' and 'test'
# =============================================================================
train = dataset.iloc[:150]
val = dataset.iloc[150:170]

train.to_csv(PROJECT_DIR + "train/train_ship.csv", index=False)
val.to_csv(PROJECT_DIR + "val/val_ship.csv", index=False)

test = test.iloc[:20]
test.to_csv(PROJECT_DIR + "test/test_ship.csv", index=False)

# =============================================================================
# Load the choosen images in the right directory
# =============================================================================
def sub_selection(dataset, origin_folder, target_folder):
    '''Copy the choosen images in the right directory'''
    filelist = list()

    for image_name in dataset.ImageId.unique():
        filelist.append(IMAGE_DIR + origin_folder + image_name)

    for file in filelist:
        shutil.copy2(file, PROJECT_DIR + target_folder)


sub_selection(train, "train_v2/", "train/")
sub_selection(val, "train_v2/", "val/")
sub_selection(test, "train_v2/", "test/")
   