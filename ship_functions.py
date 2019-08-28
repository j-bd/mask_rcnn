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
import numpy as np
import pandas as pd
import skimage.io

from mrcnn import utils
from mrcnn import model
from mrcnn import visualize
import ship_config
import ship_dataset


def create_parser():
    '''Get the informations from the operator'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command", required=True,
                        help="choice between 'train' and 'detection'")
    parser.add_argument("-of", "--origin_folder", required=True,
                        help="path to the Kaggle folder containing all data")
    parser.add_argument("-pf", "--project_folder", required=True,
                        help="path to your project folder")
    parser.add_argument("-w", "--weights", required=True,
                        help="Choice between 'coco', 'last', 'imagenet', or"\
                        "the path to a weights file usedusable by mrcnn algorithm"\
                        "to detect object")
    parser.add_argument("-sr", "--split_rate", type=float, default=0.8,
                        help="split rate between train and validation dataset during"\
                        "mrcnn training")
    args = parser.parse_args()
    return args


def structure(folder_list):
    '''Create the structure for the project and downoald necessary file'''
    for name in folder_list :
        os.makedirs(name, exist_ok=True)


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


def weights_selection(proj_dir, choice):
    '''Manage the weights file with returning the path after downloading file if
    necessary'''
        # Select weights file to load
    if choice.lower() == "coco":
        weights_path = proj_dir + "coco_trained_weights.h5"
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif choice.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif choice.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = choice

    return weights_path


def train_configuration(logs_path, weights_path, weights):
    '''Configure the MRCNN algorithme for the training'''
    config = ship_config.ShipConfig()
    config.display()

    ship_model = model.MaskRCNN(mode="training", config=config,
                                model_dir=logs_path)

    #Exclude the last COCO layers because they require a matching number of classes
    if weights.lower() == "coco":
        ship_model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
    else:
        ship_model.load_weights(weights_path, by_name=True)

    return ship_model, config


def launch_training(dataset_dir, train, val, local_model, config):
    '''Launch Mask RCNN training'''
    # Training dataset.
    dataset_train = ship_dataset.ShipDataset()
    dataset_train.load_ship(dataset_dir + "train/", train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ship_dataset.ShipDataset()
    dataset_val.load_ship(dataset_dir + "val/", val)
    dataset_val.prepare()

    # Beginning of the training
    print("Training network heads")
    local_model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=110,
                layers='heads')


def detect_configuration(result_folder, weights_path):
    '''Configure the MRCNN algorithme for the training'''
    config = ship_config.ShipConfig()
    config.display()

    ship_model = model.MaskRCNN(mode="inference", config=config,
                                model_dir=result_folder)

    ship_model.load_weights(weights_path, by_name=True)

    return ship_model, config


def mask_analyse(mask):
    '''Analyse the mrcnn mask result and return a string if exist'''
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    if mask.shape[0] > 0:
        print("[INFO]: Ship(s) detected")
        pixels = mask.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        ships = np.where(pixels[1:] != pixels[:-1])[0] + 1
        ships[1::2] -= ships[::2]
        return ' '.join(str(x) for x in ships)
    else:
        return None


def ship_detection(images_file, images_dir, local_model):
    '''Detect ships in image'''
    dataset = pd.read_csv(images_file)
    results = list()
    for image_name in dataset.iloc[:8, 0].unique():
        print("[INFO]: The following image is analysed", image_name)
        image_path = images_dir + image_name
        image = skimage.io.imread(image_path)
        res_detection = local_model.detect([image], verbose=1)[0]["masks"]
        mask_encoded = mask_analyse(res_detection)
        results.append([image_name, mask_encoded])

    return results


def export_result(list_results, folder):
    '''Export results under csv format the results'''
    dataset = pd.DataFrame(np.array(list_results), columns=['ImageId', 'EncodedPixels'])
    dataset.to_csv(folder + "submission.csv", index=False)


def display_elements(im_dir, dataset, im_index):
    '''Display the image choosen and mask (if present) with matplotlib'''
    temp_config = ship_config.ShipConfig()
    temp_dataset = ship_dataset.ShipDataset()
    temp_dataset.load_ship(im_dir, dataset)
    temp_dataset.prepare()

    image, image_meta, class_ids, bbox, mask = model.load_image_gt(temp_dataset,
                                                                   temp_config,
                                                                   im_index)

    visualize.display_instances(image, bbox, mask, class_ids, ["Ground", "Ship"])
