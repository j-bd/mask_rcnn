#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:37:18 2019

@author: j-bd

"""

import os
import argparse
import shutil

from sklearn.model_selection import train_test_split
import pandas as pd

from mrcnn import utils
from mrcnn import model
from mrcnn import visualize
import ship_config
import ship_dataset


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
                        help="split rate between train and validation dataset during"\
                        "mrcnn training")
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
    os.makedirs(train_images_dir + 'train', exist_ok=True)
    os.makedirs(train_images_dir + 'val', exist_ok=True)
    os.makedirs(backup, exist_ok=True)

#A integrer dans def configuration dependemment du choix :
# coco, imagenet, reprendre a partir d'un autre fichier
    weights_path = proj_dir + "coco_trained_weights.h5"
    utils.download_trained_weights(weights_path)

    return weights_path

def images_transfert(dataset, o_folder, f_folder):
    '''Copy selected images from a folder to another one'''
    filelist = list()
    for image_name in dataset.ImageId.unique():
        filelist.append(o_folder + image_name)
    print(f"[INFO] Copying of {len(filelist)} images from {o_folder} to "\
          f"{f_folder} on progress. Please wait.")
    for file in filelist:
        shutil.copy2(file, f_folder)
    print("[INFO] Copy is done.")

def images_copy(csv_file, o_folder, p_folder, percent_images=1, val_size=0.2):
    '''Copy images for the training with respected proportion
    dataset: A pandas dataset is provided,
    percent_images: A percent of images to copy from Kaggle folder to project,
    val_size: percent of val images,
    o_folder: path to the origin folder,
    p_folder: path to the project folder'''
    dataset = pd.read_csv(csv_file)
    dataset["Mask"] = dataset["EncodedPixels"].apply(
        lambda x: 0 if isinstance(x, float) else 1)
    X = dataset.iloc[:, :2]
    y = dataset.iloc[:, -1]
    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      train_size=percent_images,
                                                      test_size=val_size*percent_images,
                                                      random_state=42,
                                                      stratify=y)
    images_transfert(X_train, o_folder, p_folder + "train/")
    images_transfert(X_val, o_folder, p_folder + "val/")

    return X_train, X_val

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


def display_elements(im_dir, dataset, im_index):
    '''Display the image choosen with matplotlib'''
    temp_config = ship_config.ShipConfig()
    temp_dataset = ship_dataset.ShipDataset()
    temp_dataset.load_ship(im_dir, dataset)
    temp_dataset.prepare()

    image, image_meta, class_ids, bbox, mask = model.load_image_gt(temp_dataset,
                                                                   temp_config,
                                                                   im_index)

    visualize.display_instances(image, bbox, mask, class_ids, ["Ground", "Ship"])
