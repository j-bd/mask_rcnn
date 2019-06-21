{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Implementation of Mask RCNN\n",
    "\n",
    "You will find here a test application of M-RCNN deep learning method to an images detection challenge."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data source\n",
    "\n",
    "I used data made available by airbus on Kaggle. The challenge name is [\"Airbus Ship Detection Challenge\"](https://www.kaggle.com/c/airbus-ship-detection).\n",
    "\n",
    "The second version of data is organised as :\n",
    "    - A train folder with about 192 550 images,\n",
    "    - A test folder with aout 15 600 images,\n",
    "    - A csv file named \"train_ship_segmentations_v2.csv\" with the mask of all train images,\n",
    "    - A csv file named \"sample_submission_v2.csv\" to show the shape of the output expected."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Global organisation of the repository\n",
    "\n",
    "The folder **mrcnn** gather the heart of the Mask RCNN functioning. It comes from the great work of [Mask R-CNN library](https://github.com/matterport/Mask_RCNN) that Matterport built.\n",
    "\n",
    "The files ```learning_ship.py``` and ```ship_detection.py``` are mostly from the work of [gabrielgarza](https://github.com/gabrielgarza/Mask_RCNN/tree/master/samples/ship)\n",
    "\n",
    "Finally, the file ```data_preprocessing.py``` is the one I did in order to select few images in each Airbus *test* and *train* folders. Indeed, as I worked on a weak computer, I could not allow to train the model on the full data. Anyway, the aim was to understand the full process."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
