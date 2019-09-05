#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:05:17 2019

@author: j-bd
"""
import os

import numpy as np
from mrcnn import utils

import constants


class ShipDataset(utils.Dataset):
    '''class to manage the data for the training process with mrcnn'''
    def load_ship(self, dataset_dir, subset):
        ''' Reads pandas dataframe, extracts the annotations, and iteratively
        calls the internal 'add_class' and 'add_image' functions to build the
        dataset.
        img_masks is a list with this format ['183672 1 184439 2 185206 4']'''
        self.add_class("ship", 1, "ship")

        for image in subset.ImageId.unique():
            img_masks = subset.loc[subset['ImageId'] == image,
                                   'EncodedPixels'].tolist()
            self.add_image(
                "ship", image_id=image, path=os.path.join(dataset_dir, image),
                width=constants.WIDTH, height=constants.HEIGHT, img_masks=img_masks
            )

    def mask_creation(self, mask_encode, shape):
        '''Create a mask from airbus string information to a numpy array
        The original information given are by column then row. Here 'mask_encode'
        is a str() with the following format : 1 3 10 5 ...
        It's work by even that contain a start position and a run length.
        E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3)
        '''
        img = np.zeros(np.prod(shape), dtype=np.uint8)
        # If there is no mask on image (empty value for mask_encode)
        if not isinstance(mask_encode, str):
            return img.reshape(shape)

        # For images with mask_encode (mean presence of object) we need to mark
        # '1' instead of '0' in the array 'img' at the position of the mask.
        lines = len(mask_encode.split())
        starts, lengths = np.hsplit(
            np.fromstring(mask_encode, dtype=int, sep=' ').reshape(lines, 2), 2
        )
        starts -= 1
        ends = starts + lengths
        for start, end in zip(starts, ends):
            img[start:end] = 1
        return img.reshape(shape).T


    def load_mask(self, image_id):
        '''Generates bitmap masks for every object in the image by drawing the
        polygons.'''
        info = self.image_info[image_id]
        # If not a ship dataset image, delegate to parent class.
        if info["source"] != "ship":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        shape = [info["height"], info["width"]]
        mask_array = np.zeros(
            [info["height"], info["width"], len(info["img_masks"])],
            dtype=np.uint8
        )

        for index, mask in enumerate(info["img_masks"]):
            mask_array[:, :, index] = self.mask_creation(mask, shape)

        # Mask + class of the Mask (here, we have only one class)
        return mask_array.astype(np.bool), np.ones([mask_array.shape[-1]], dtype=np.int32)
