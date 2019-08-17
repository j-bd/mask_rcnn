#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:37:18 2019

@author: j-bd

"""

import os
import argparse

from mrcnn import utils
from mrcnn import model
import ship_config


def create_parser():
    '''Get the informations from the operator'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training",
                        help="command to prepare data in order to lauch mrcnn training")
    parser.add_argument("-of", "--origin_folder", required=True,
                        help="path to the Kaggle folder containing all data")
    parser.add_argument("-pf", "--project_folder", required=True,
                        help="path to your project folder")
#    parser.add_argument("-b", "--batch", type=int, default=64,
#                        help="batch number for yolo config file used during yolo training")
#    parser.add_argument("-s", "--subdivisions", type=int, default=16,
#                        help="subdivisions number for yolo config file used during yolo training")
    parser.add_argument("-sr", "--split_rate", type=float, default=0.8,
                        help="split rate between train and validation dataset during mrcnn training")
    parser.add_argument("-d", "--detection",
                        help="command to detect pneumonia object on image")
    parser.add_argument("-w", "--weights_path",
                        help="Path to the weights file used by mrcnn algorithm to detect object")
#    parser.add_argument("-c", "--confidence", type=float, default=0.2,
#                        help="minimum probability to filter weak detections")
#    parser.add_argument("-t", "--threshold", type=float, default=0.2,
#                        help="threshold when applying non-maxima suppression")
#    parser.add_argument("-dis", "--detect_im_size", type=int, default=640,
#                        help="resize input image to improve the detection"\
#                        "(must be a multiple of 32)")
    args = parser.parse_args()
    return args

def structure(proj_dir, train_images_dir, backup):
    '''Create the structure for the project and downoald necessary file'''
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(backup, exist_ok=True)

    print(f"[INFO] Please, clone mrcnn repository in '{proj_dir}' if necessary.")

#A integrer dans def configuration dependemment du choix : coco, imagenet, reprendre a partir d'un autre fichier
    weights_path = proj_dir + "coco_trained_weights.h5"
    utils.download_trained_weights(weights_path)

    return weights_path

def configuration(logs_path, weights_path):
    '''Configure the MRCNN algorithme for the training'''
    config = ship_config.ShipConfig()
    config.display()

    ship_model = model.MaskRCNN(mode="training", config=config,
                                model_dir=logs_path)

    #Seulement vrai pour COCO. A faire pour les autres
    #Exclude the last layers because they require a matching number of classes
    ship_model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

    return ship_model
