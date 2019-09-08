#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:37:18 2019

@author: j-bd

"""

import os
import argparse
import shutil
import logging

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import skimage.io
from mrcnn import utils
from mrcnn import model
from mrcnn import visualize

import ship_config
import ship_dataset


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

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
                        help="Choice between 'coco', 'last', 'imagenet', or "\
                        "the path to a weights file usedusable by mrcnn algorithm "\
                        "to detect object")
    parser.add_argument("-iq", "--images_quantity", type=float, default=0.02,
                        help="Quantity of images selected in the folder for the training "\
                        "Must be 0 < images_quantity <= 1")
    parser.add_argument("-sr", "--split_rate", type=float, default=0.8,
                        help="split rate between train and validation dataset during "\
                        "mrcnn training. Must be 0.6 < images_quantity <= 0.95")
    parser.add_argument("-en", "--epoch_number", default=110,
                        help="total number of iteration")
    args = parser.parse_args()
    return args


def check_input(args):
    '''Check if inputs are correct'''
    if args.command not in ["train", "detection"]:
        raise ValueError(
            "Your choice for '-c', '--command' must be 'train' or 'detection'."
        )
    if not os.path.isdir(args.origin_folder):
        raise FileNotFoundError(
            "Your choice for '-of', '--origin_folder' is not a valide directory."
        )
    if not os.path.isdir(args.project_folder):
        raise FileNotFoundError(
            "Your choice for '-of', '--project_folder' is not a valide directory."
        )
    if not 0.6 <= args.split_rate <= 0.95:
        raise ValueError(
            f"Split rate must be between 0,6 and 0.95, currently {args.split_rate}."
        )
    weights = args.weights
    if not weights.endswith(("coco", "last", "imagenet", ".h5")):
        raise FileNotFoundError(
            "Your choice for '-w', '--weights' must be between 'coco', 'last', 'imagenet', or "\
            "the path to a weights file usedusable by mrcnn algorithm to detect object."
        )
    if not os.path.isdir(os.path.join(args.project_folder, "mrcnn")):
        raise FileNotFoundError(
            f"Please, clone mrcnn repository in '{args.project_folder}'."
        )


def structure(folder_list):
    '''Create the structure for the project and download necessary file'''
    for name in folder_list:
        os.makedirs(name, exist_ok=True)


def images_transfert(df, o_folder, f_folder):
    '''Copy selected images from a folder to another one'''
    filelist = list()
    for image_name in df.ImageId.unique():
        filelist.append(os.path.join(o_folder, image_name))
    logging.info(f" Copying of {len(filelist)} images from {o_folder} to "\
          f"{f_folder} on progress. Please wait.")
    for file in filelist:
        shutil.copy2(file, f_folder)
    logging.info(" Copy is done.")


def images_copy(csv_file, o_folder, p_folder, percent_images=1, val_size=0.2):
    '''Copy images for the training with respected proportion
    df: A pandas DataFrame is provided,
    percent_images: A percent of images to copy from Kaggle folder to project,
    val_size: percent of val images,
    o_folder: path to the origin folder,
    p_folder: path to the project folder'''
    df = pd.read_csv(csv_file)
    df["Mask"] = df["EncodedPixels"].apply(
        lambda x: 0 if isinstance(x, float) else 1
    )
    xs = df.iloc[:, :2]
    ys = df.iloc[:, -1]
    x_train, x_val, y_train, y_val = train_test_split(
        xs, ys, train_size=percent_images, test_size=val_size*percent_images,
        random_state=42, stratify=ys
    )
    images_transfert(x_train, o_folder, os.path.join(p_folder, "train"))
    images_transfert(x_val, o_folder, os.path.join(p_folder, "val"))

    return x_train, x_val


def weights_selection(proj_dir, choice):
    '''Manage the weights file with returning the path after downloading file if
    necessary'''
        # Select weights file to load
    if choice.lower() == "coco":
        weights_path = os.path.join(proj_dir, "coco_trained_weights.h5")
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
    '''Configure the MRCNN algorithm for the training'''
    config = ship_config.ShipConfig()
    config.display()

    ship_model = model.MaskRCNN(
        mode="training", config=config, model_dir=logs_path
    )

    # Exclude the last COCO layers because they require a matching number of classes
    if weights.lower() == "coco":
        ship_model.load_weights(
            weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                                 "mrcnn_bbox", "mrcnn_mask"]
        )
    else:
        ship_model.load_weights(weights_path, by_name=True)

    return ship_model, config


def launch_training(dataset_dir, train, val, local_model, config, args):
    '''Launch Mask RCNN training'''
    # Training dataset.
    dataset_train = ship_dataset.ShipDataset()
    dataset_train.load_ship(os.path.join(dataset_dir, "train"), train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ship_dataset.ShipDataset()
    dataset_val.load_ship(os.path.join(dataset_dir, "val"), val)
    dataset_val.prepare()

    # Beginning of the training
    print("Training network heads")
    local_model.train(
        dataset_train, dataset_val, learning_rate=config.LEARNING_RATE,
        epochs=args.epoch_number, layers='heads'
    )


def detect_configuration(result_folder, weights_path):
    '''Configure the MRCNN algorithm for the training'''
    config = ship_config.ShipConfig()
    config.display()

    ship_model = model.MaskRCNN(
        mode="inference", config=config, model_dir=result_folder
    )

    ship_model.load_weights(weights_path, by_name=True)

    return ship_model, config


def mask_analyse(mask):
    '''Analyse the mrcnn mask result and return a string if exist.
    mask: numpy array with True False data. E.g [False False True True False ...]
    output: The string format must be by even that contain a start position and
    a run length. E.g. '1 3 8 2' implies starting at pixel 1 and running a total
    of 3 pixels (1,2,3) then starting at pixel 8 and running a total of 2 pixels
    (8, 9)
    '''
    # We're treating all instances as one, so collapse the mask into one layer
    print("mask before:", type(mask), mask.shape, mask)
    mask = mask.any(axis=-1, keepdims=True)

    if mask.any():
        logging.info(" Ship(s) detected")
        print("mask:", type(mask), mask.shape, mask)
        pixels = mask.T.flatten()

#        pixels = np.r_[0, pixels, 0]
#        ships = np.where(pixels[1:] != pixels[:-1])[0] + 1
        # Calculate the length and overwrite the result on the "end" location
#        ships[1::2] -= ships[::2]

        # Get an array with the indexs of biginnings and ends of 1
        ships = np.nonzero(np.ediff1d(pixels, to_end=0, to_begin=0))[0] + 1
        lengths = np.diff(np.hsplit(ships, 2))
        starts = np.delete(np.hsplit(ships, 2), 1, axis=1)
        # Get an array as [index_beginning length index_beginning length ...]
        ships = np.concatenate((starts, lengths), axis=1).flatten()

        return ' '.join(str(x) for x in ships)
    else:
        return None


def ship_detection(images_file, images_dir, local_model):
    '''Detect ships in image'''
    df = pd.read_csv(images_file)
    count = 1
    results = list()
    for image_name in df.iloc[:, 0].unique():
        logging.info(
            f" {count} / {len(df.iloc[:, 0].unique())}. "
            f"The following image is analysed: {image_name}"
        )
        image_path = os.path.join(images_dir, image_name)
        image = skimage.io.imread(image_path)
        res_detection = local_model.detect([image], verbose=0)[0]["masks"]
        mask_encoded = mask_analyse(res_detection)
        results.append([image_name, mask_encoded])
        count += 1

    return results


def export_result(list_results, folder):
    '''Export results under csv format the results'''
    df = pd.DataFrame(np.array(list_results), columns=['ImageId', 'EncodedPixels'])
    df.to_csv(os.path.join(folder, "submission.csv"), index=False)


def display_elements(im_dir, dataset, im_index):
    '''Display the image choosen and mask (if present) with matplotlib'''
    temp_config = ship_config.ShipConfig()
    temp_dataset = ship_dataset.ShipDataset()
    temp_dataset.load_ship(im_dir, dataset)
    temp_dataset.prepare()

    image, image_meta, class_ids, bbox, mask = model.load_image_gt(
        temp_dataset, temp_config, im_index
    )

    visualize.display_instances(image, bbox, mask, class_ids, ["Ground", "Ship"])
