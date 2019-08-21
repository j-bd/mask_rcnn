#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:17:16 2019

@author: j-bd

Implementing Mask RCNN for our own dataset:
Step 1: Clone Mask RCNN repository
Step 2: As mentionned by authors, we need first to create a sub-class config
based on mrcnn/config.py. It will be not automatically implemented here
Step 3: We create the structure of the project. We need:
    A master project folder (PROJECT_DIR),
    A sub-folder 'train' with all training images (automatically recogmised),
    A sub-folder 'val' with all validation images (automatically recogmised),
    A sub-folder 'test' with all test images,
    A sub-folder that will gather training logs during proceding (BACKUP).
Step 4: We need a pre-trained weights file. If we choose Coco or Imagenet, we can
download it with provided commands (thanks !). We can also used a last file (for
instance, if we want to resume a training)
Step 5: We organised images between "train" and "val" (validation) folders. We
offer the possibility to choose the amount of data while respecting the ratio
between images with and without mask
Step 6: We charge the configuration file create in step 2
Step 7: We instantiate the MaskRCNN Class present in mrcnn/model.py
Step 8: We load the pre-trained weights file. See step 4
Step 9: We instantiate the ShipDataset class (based on Dataset class in
mrcnn/utils.py) for:
    managing training data
    managing validation data
"""

import ship_functions


def pre_trainning():#args + renommer
    '''Lauch all necessary steps to set up Mask RCNN algorithm before trainning'''
    ORIGIN_DIR = "/media/latitude/TOSHIBA EXT/20180831-Sauvegarde_Toshiba/E/Prog/Kaggle/airbus_ship/airbus-ship-detection/data/"
    ORIGIN_TRAIN_DIR = ORIGIN_DIR + "train_v2/"
    FILE_DESCR = "train_ship_segmentations_v2.csv"
    ORIGIN_TEST_DIR = ORIGIN_DIR + "test_v2/"
    PROJECT_DIR = "/home/latitude/Documents/Kaggle/airbus_ship/m_rcnn/"
#    TRAIN_DATA_DIR = PROJECT_DIR + "data/"
    TRAIN_IMAGES_DIR = PROJECT_DIR + "data/"
    BACKUP = PROJECT_DIR + "backup_log/"
#    FILE_TRAIN = "stage_2_train_labels.csv"
#    IMAGE_SIZE = 1024
#    OBJ_NBR = 1
#    TEST_IMAGES_DIR = PROJECT_DIR + "detect_results/obj/"

    input(f"[INFO] Please, clone mrcnn repository in '{PROJECT_DIR}' if "\
          "necessary. Once it is done, please press 'enter'")

    weight_path = ship_functions.structure(PROJECT_DIR, TRAIN_IMAGES_DIR, BACKUP)
    train_data, val_data = ship_functions.images_copy(ORIGIN_DIR + FILE_DESCR,
                                                      ORIGIN_TRAIN_DIR,
                                                      TRAIN_IMAGES_DIR,
                                                      percent_images=0.02,
                                                      val_size=0.2)

    ship_functions.configuration(BACKUP, weight_path)

def main():
    '''Allow the selection between algorithm training or image detection'''
#    args = ship_functions.create_parser()
#    il args.choix = training:
    pre_trainning()
#    else: (detection)

if __name__ == "__main__":
    main()
