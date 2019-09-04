#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:05:17 2019

@author: j-bd
"""
import os

import numpy as np

from mrcnn import utils


class ShipDataset(utils.Dataset):
    '''class to manage the data for the training process with mrcnn'''
    def load_ship(self, dataset_dir, subset):
        ''' Reads pandas dataframe, extracts the annotations, and iteratively
        calls the internal 'add_class' and 'add_image' functions to build the
        dataset.'''
        self.add_class("ship", 1, "ship")

        for image in subset.ImageId.unique():
            img_masks = subset.loc[subset['ImageId'] == image,
                                   'EncodedPixels'].tolist()
            self.add_image("ship",
                           image_id=image,
                           path=os.path.join(dataset_dir, image),
                           width=768,
                           height=768,
                           img_masks=img_masks)


    def mask_creation(self, mask_encode, shape=(768, 768)):
        '''Create a mask from airbus string information to a numpy array'''
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        if not isinstance(mask_encode, str):
            return img.reshape(shape)

        mask_split = mask_encode.split()
        starts, lengths = [np.asarray(
            x, dtype=int) for x in (mask_split[0::2], mask_split[1::2])]
        starts -= 1
        ends = starts + lengths
        for start, end in zip(starts, ends):
            img[start:end] = 1
        return img.reshape(shape).T


    def load_mask(self, image_id):
        '''Generates bitmap masks for every object in the image by drawing the
        polygons.'''
        # If not a ship dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "ship":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        info = self.image_info[image_id]
        shape = [info["height"], info["width"]]

        mask_array = np.zeros(
            [info["height"], info["width"], len(info["img_masks"])],
            dtype=np.uint8)

        for index, mask in enumerate(info["img_masks"]):
            mask_array[:, :, index] = self.mask_creation(mask, shape)

        # Mask + class of the Mask (here, we have only one class)
        return mask_array.astype(np.bool), np.ones([mask_array.shape[-1]], dtype=np.int32)
