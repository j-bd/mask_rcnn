#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:17:16 2019

@author: j-bd

"""

import os

import ship_functions


def detection(args):
    '''Launch the detection for selected images'''
    origin_dir = args.origin_folder
    origin_test_dir = os.path.join(origin_dir, "test_v2")
    file_cont = os.path.join(origin_dir, "sample_submission_v2.csv")
    project_dir = args.project_folder
    detect = os.path.join(project_dir, "results")

    ship_functions.structure([project_dir, detect])
    ship_model, config = ship_functions.detect_configuration(detect, args.weights)
    result = ship_functions.ship_detection(
        file_cont, origin_test_dir, ship_model
    )
    ship_functions.export_result(result, detect)


def training(args):
    '''Lauch all necessary steps from Mask RCNN algorithm set up to the trainning'''
    origin_dir = args.origin_folder
    origin_train_dir = os.path.join(origin_dir, "train_v2")
    file_descr = os.path.join(origin_dir, "train_ship_segmentations_v2.csv")
    project_dir = args.project_folder
    train_image_dir = os.path.join(project_dir, "data")
    backup = os.path.join(project_dir, "backup_log")

    ship_functions.structure(
        [project_dir, train_image_dir, backup, os.path.join(train_image_dir, 'train'),
         os.path.join(train_image_dir, 'val')]
    )

    train_data, val_data = ship_functions.images_copy(
        file_descr, origin_train_dir, train_image_dir, percent_images=0.02,
        val_size=1 - args.split_rate
    )

    weight_path = ship_functions.weights_selection(project_dir, args.weights)

    ship_model, config = ship_functions.train_configuration(
        backup, weight_path, args.weights
    )

    ship_functions.launch_training(
        train_image_dir, train_data, val_data, ship_model, config
    )


def main():
    '''Allow the selection between training algorithm or image detection'''
    args = ship_functions.create_parser()
    ship_functions.check_input(args)

#    if args.command == "train":
#        training(args)
#    else:
#        detection(args)


if __name__ == "__main__":
    main()
