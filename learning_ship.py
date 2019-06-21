#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:19:56 2019

@author: https://github.com/gabrielgarza/Mask_RCNN/blob/master/samples/ship/ship.py
        optimised by j-bd

Mask R-CNN
Train on the toy Ship dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 learning_ship.py train --dataset=data --weights=coco
    # Resume training a model that you had trained earlier
    python3 ship.py train --dataset=./datasets --weights=../../logs/../mask_rcnn_ship_0067.h5
    # Train a new model starting from ImageNet weights
    python3 ship.py train --dataset=./datasets --weights=imagenet
"""

import os
import sys

import numpy as np
import pandas as pd
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


# =============================================================================
# Configurations
# =============================================================================
class ShipConfig(Config):
    """
    Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ship"

    # As I use a laptop, no much computing power available
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + ship

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 250

    # Skip detections with < 95% confidence
    DETECTION_MIN_CONFIDENCE = 0.95

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.0

    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768


# =============================================================================
# Dataset
# =============================================================================

class ShipDataset(utils.Dataset):
    """
    Method to manage the training with a subset of data and mask
    """
    def load_ship(self, dataset_dir, subset):
        """
        Load a subset of the Ship dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("ship", 1, "ship")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load image ids (filenames) and run length encoded pixels
        ship_seg_df = pd.read_csv(os.path.join(dataset_dir,
                                               "{}_ship_segmentations.csv".format(subset)))
        ship_seg_df = ship_seg_df.sample(frac=0.05)
        unique_image_ids = ship_seg_df.ImageId.unique()

        for image_id in unique_image_ids:
            img_masks = ship_seg_df.loc[
                ship_seg_df['ImageId'] == image_id, 'EncodedPixels'
            ].tolist()
            image_path = os.path.join(dataset_dir, image_id)

            if os.path.isfile(image_path):
                self.add_image(
                    "ship",
                    image_id=image_id,  # use file name as a unique image id
                    path=image_path,
                    width=768, height=768,
                    img_masks=img_masks)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a ship dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "ship":
            return super(self.__class__, self).load_mask(image_id)

        # Convert RLE Encoding to bitmap mask of shape [height, width, instance count]
        info = self.image_info[image_id]
        img_masks = info["img_masks"]
        shape = [info["height"], info["width"]]

        # Mask array placeholder
        mask_array = np.zeros(
            [info["height"], info["width"], len(info["img_masks"])], dtype=np.uint8)

        # Build mask array
        for index, mask in enumerate(img_masks):
            mask_array[:, :, index] = self.rle_decode(mask, shape)

        return mask_array.astype(np.bool), np.ones([mask_array.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ship":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def rle_encode(self, img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def rle_decode(self, mask_rle, shape=(768, 768)):
        '''
        Convert a Run Length Encoded Mask (provide by airbus) to an image mask
        mask_rle: run-length as string formated (start length) according to the provider
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        '''
        if not isinstance(mask_rle, str):
            img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
            return img.reshape(shape).T

        mask_split = mask_rle.split()
        starts, lengths = [np.asarray(
            x, dtype=int) for x in (mask_split[0:][::2], mask_split[1:][::2]
                                    )]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for start, end in zip(starts, ends):
            img[start:end] = 1
        return img.reshape(shape).T


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ShipDataset()
    dataset_train.load_ship(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ShipDataset()
    dataset_val.load_ship(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=110,
                layers='heads')


# =============================================================================
# Training
# =============================================================================

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect ships.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/ship/dataset/",
                        help='Directory of the Ship dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ShipConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. ""Use 'train'".format(args.command))
