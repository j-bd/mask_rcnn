#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:37:18 2019

@author: j-bd

Step 1: As mentionned by authors, we need first to create a sub-class config
based on mrcnn/config.py
"""

import os

from mrcnn import utils


def structure(proj_dir, train_images_dir, backup):
    '''Create the structure for the project and downoald necessary file'''
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(backup, exist_ok=True)

    print(f"[INFO] Please, clone mrcnn repository in '{proj_dir}' if necessary.")

    utils.download_trained_weights(proj_dir + "coco_trained_weights.h5")
