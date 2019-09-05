#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:17:19 2019

@author: j-bd

"""

from mrcnn.config import Config

import constants

class ShipConfig(Config):
    """
    Configuration for training on the ship dataset.
    Derives from the base Config class and overrides some values as recommanded
    by author in the clone file "mrcnn/config.py".
    """
    NAME = "ship"

    #CPU used. Value to "1" as recommanded
    IMAGES_PER_GPU = 1

    # Number of classes (Background + ship)
    NUM_CLASSES = 2

    STEPS_PER_EPOCH = constants.STEPS_PER_EPOCH

    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = constants.DETECTION_MIN_CONFIDENCE

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = constants.DETECTION_NMS_THRESHOLD

    IMAGE_MIN_DIM = constants.HEIGHT
    IMAGE_MAX_DIM = constants.WIDTH
