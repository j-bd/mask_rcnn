#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:17:16 2019

@author: j-bd

To launch the training process:
python main.py -c train -of "/path/to/kaggle/data/folder/"
-pf "/path/to/your/project/folder/" -w "/path/to/your/file.h5"
All these arguments are mandatory. Regarding the weights choice "-w". You either
can provide:
    the path to a file "wights.h5"
    or enter "coco" or "imagenet" to make automaticaly downloading

To lauch detection:
python main.py -c detection -of "/path/to/kaggle/data/folder/"
 -pf "/path/to/your/project/folder/" -w "/path/to/your/file.h5"
All these arguments are mandatory.
"""

import ship_functions


def detection(args):
    '''Launch the detection for selected images'''
    ORIGIN_DIR = args.origin_folder
    ORIGIN_TEST_DIR = ORIGIN_DIR + "test_v2/"
    FILE_CONT = "sample_submission_v2.csv"
    PROJECT_DIR = args.project_folder
    DETECTION = PROJECT_DIR + "results/"

    ship_functions.structure([PROJECT_DIR, DETECTION])
    ship_model, config = ship_functions.detect_configuration(DETECTION, args.weights)
    result = ship_functions.ship_detection(ORIGIN_DIR + FILE_CONT,
                                  ORIGIN_TEST_DIR,
                                  ship_model)
    ship_functions.export_result(result, DETECTION)


def trainning(args):
    '''Lauch all necessary steps to set up Mask RCNN algorithm before trainning'''
    ORIGIN_DIR = args.origin_folder
    ORIGIN_TRAIN_DIR = ORIGIN_DIR + "train_v2/"
    FILE_DESCR = "train_ship_segmentations_v2.csv"
    PROJECT_DIR = args.project_folder
    TRAIN_IMAGES_DIR = PROJECT_DIR + "data/"
    BACKUP = PROJECT_DIR + "backup_log/"

    input(f"[INFO] Please, clone mrcnn repository in '{PROJECT_DIR}' if "\
          "necessary. Once it is done, please press 'enter'")

    ship_functions.structure([PROJECT_DIR,
                             TRAIN_IMAGES_DIR,
                             BACKUP,
                             TRAIN_IMAGES_DIR + 'train',
                             TRAIN_IMAGES_DIR + 'val'])

    train_data, val_data = ship_functions.images_copy(ORIGIN_DIR + FILE_DESCR,
                                                      ORIGIN_TRAIN_DIR,
                                                      TRAIN_IMAGES_DIR,
                                                      percent_images=0.02,
                                                      val_size=1-args.split_rate)

    weight_path = ship_functions.weights_selection(PROJECT_DIR, args.weights)

    ship_model, config = ship_functions.train_configuration(BACKUP,
                                                            weight_path,
                                                            args.weights)

    ship_functions.launch_training(TRAIN_IMAGES_DIR,
                                   train_data,
                                   val_data,
                                   ship_model,
                                   config)


def main():
    '''Allow the selection between training algorithm or image detection'''
    args = ship_functions.create_parser()
    if args.command == "train":
        trainning(args)
    elif args.command == "detection":
        detection(args)
    else:
        print("[INFO] Your choice for '-c', '--command' must be 'train' or 'detection'")


if __name__ == "__main__":
    main()
