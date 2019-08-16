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
from mrcnn import model
import ship_config


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

def configuration(logs_path, weights_path, ):
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