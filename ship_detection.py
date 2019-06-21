#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:37:26 2019

@author: https://github.com/gabrielgarza/Mask_RCNN/blob/master/samples/ship/generate_predictions.py
        modifier by j-bd
"""

import os
import sys
import time
import datetime

import argparse
import pandas as pd
import numpy as np
import skimage.io

import mrcnn.model as modellib
import learning_ship

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# To find local version of the library
sys.path.append(ROOT_DIR)


if __name__ == '__main__':


    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Predict Mask R-CNN to detect ships.')

    parser.add_argument('--weights', required=True,
                        metavar="path/to/log/file.h5",
                        help="Path to weights .h5 file")
    args = parser.parse_args()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Path to trained weights
    SHIP_WEIGHTS_PATH = args.weights

    # Config
    config = learning_ship.ShipConfig()
    SHIP_DIR = os.path.join(ROOT_DIR, "data")

    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        '''
        Configuration to run detection on one image at a time
        '''
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.95
        DETECTION_NMS_THRESHOLD = 0.0
        IMAGE_MIN_DIM = 768
        IMAGE_MAX_DIM = 768
        RPN_ANCHOR_SCALES = (64, 96, 128, 256, 512)
        DETECTION_MAX_INSTANCES = 20

    # Create model object in inference mode.
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Instantiate dataset
    dataset = learning_ship.ShipDataset()

    # Load weights
    model.load_weights(os.path.join(ROOT_DIR, SHIP_WEIGHTS_PATH), by_name=True)
    class_names = ['BG', 'ship']

    # Run detection
    # Load image ids (filenames) and run length encoded pixels
    images_path = "path/to/test_pictures/folder"
    sample_sub_csv = "name_of_csv_file/in/images_path/folder"
    sample_submission_df = pd.read_csv(os.path.join(images_path, sample_sub_csv))
    unique_image_ids = sample_submission_df.ImageId.unique()

    out_pred_rows = []
    count = 0
    for image_id in unique_image_ids:
        image_path = os.path.join(images_path, image_id)
        if os.path.isfile(image_path):
            count += 1
            print("Step: ", count)

            # Start counting prediction time
            tic = time.clock()

            image = skimage.io.imread(image_path)
            results = model.detect([image], verbose=1)
            r = results[0]

            # First Image
            re_encoded_to_rle_list = []
            for i in np.arange(np.array(r['masks']).shape[-1]):
                boolean_mask = r['masks'][:, :, i]
                re_encoded_to_rle = dataset.rle_encode(boolean_mask)
                re_encoded_to_rle_list.append(re_encoded_to_rle)

            if len(re_encoded_to_rle_list) == 0:
                out_pred_rows += [{'ImageId': image_id, 'EncodedPixels': None}]
                print("Found Ship: ", "NO")
            else:
                for rle_mask in re_encoded_to_rle_list:
                    out_pred_rows += [{'ImageId': image_id, 'EncodedPixels': rle_mask}]
                    print("Found Ship: ", rle_mask)
            toc = time.clock()
            print("Prediction time: ", toc-tic)

    submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]

    filename = "{}{:%Y%m%dT%H%M}.csv".format("submissions/submission_", datetime.datetime.now())
    submission_df.to_csv(filename, index=False)

    print("Submission CSV Shape", submission_df.shape)
