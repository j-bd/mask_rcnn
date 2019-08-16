#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:17:16 2019

@author: j-bd
"""

import ship_functions
import ship_config


def pre_trainning():#args
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

    weight_path = ship_functions.structure(PROJECT_DIR,TRAIN_IMAGES_DIR, BACKUP)

    ship_functions.configuration(weight_path, BACKUP)

def main():
    '''Allow the selection between algorithm training or image detection'''
    pre_trainning()

if __name__ == "__main__":
    main()
