#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:26:57 2019

@author: latitude
"""

IMAGES_PER_GPU = 1
# Number of training steps per epoch
STEPS_PER_EPOCH = 250
# Minimum probability value to accept a detected instance
DETECTION_MIN_CONFIDENCE = 0.9
# Non-maximum suppression threshold for detection
DETECTION_NMS_THRESHOLD = 0.2
# Image maximum dimension
WIDTH = 768
# Image minimum dimension
HEIGHT = 768
