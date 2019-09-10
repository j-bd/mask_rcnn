

# Implementation of Mask RCNN

You will find here a test application of M-RCNN deep learning method to an images detection challenge.


## Overview of Mask-RCNN pipeline

![Image of mrcnn](https://cdn-images-1.medium.com/max/2600/1*M_ZhHp8OXzWxEsfWu2e5EA.png)

Mask-RCNN [paper](https://arxiv.org/pdf/1703.06870.pdf) were submit on 20 Mar 2017


## Installation

Please, give a look to this author [page](https://github.com/matterport/Mask_RCNN#installation) to know how to install Mask RCNN.


## Data source

I used data made available by airbus on Kaggle. The challenge name is ["Airbus Ship Detection Challenge"](https://www.kaggle.com/c/airbus-ship-detection).

The second version of data is organised as :
* A train folder with about 192 550 images,
* A test folder with about 15 600 images,
* A csv file named "train_ship_segmentations_v2.csv" with the mask of all train images,
* A csv file named "sample_submission_v2.csv" to show the shape of the output expected.


## Implementing Mask RCNN for our own dataset and process for the training:

The following steps are necessary:
* Step 1: Clone [Mask RCNN repository](https://github.com/matterport/Mask_RCNN)
* Step 2: As mentionned by authors, we need first to create a sub-class config
based on ```mrcnn/config.py```. **It will be not automatically implemented here**. Thus, we create an inherit class and customed it following our computer capacity. This file is ```ship_config.py```
* Step 3: As also mentionned by authors, we need to customize the ```class Dataset``` based on the one present in algorithm ```utils.py```. **It will be not automatically implemented here**.Thus, we create an inherit class and customed it in order to manage images and the mask linked to. This file is ```ship_dataset.py```
* Step 4: We create the structure of the project. We need:
   * A master project folder (PROJECT_DIR),
   * A sub-folder 'train' with all training images,
   * A sub-folder 'val' with all validation images,
   * A sub-folder 'backup_log' that will gather training logs during proceding (BACKUP).
* Step 5: We need a pre-trained weights file. If we choose Coco or Imagenet, we can
download it with provided commands (thanks to thte mrcnn authors!). We need to mention it when launching the training. We can also use a last file (for instance, if we want to resume a training)
* Step 6: We organised images between "train" and "val" (validation) folders. We
offer the possibility to choose the amount of data while respecting the ratio
between images with and without mask
* Step 7: We charge the configuration file create in step 2
* Step 8: We instantiate the MaskRCNN Class present in mrcnn/model.py
* Step 9: We load the pre-trained weights file. See step 4
* Step 10: We instantiate the ShipDataset class.
* Step 11: Training can be launch

**Two parameters can be modulated for the training** :
* Numbers of steps per epoch in ship_config.py
* Numbers of epochs in ship_functions.py


## Before any code execution

Before launching this algorithm make sure the Kaggle data are organized as following in a master directory:
* A directory with your images test named 'test_v2',
* A directory with your images train named 'train_v2',
* A detailed CSV file train labels named 'train_ship_segmentations_v2',
* A detailed CSV file for submission named 'sample_submission_v2'.

All this elements must be gathered in the same directory. The path will be mentionned when launching the algorithm.


## Terminal commands

For all commands, please make sure to be located in the project folder.

To launch the training process:
* ```python main.py --command train --origin_folder "/path/to/kaggle/data/folder/" --project_folder "/path/to/your/project/folder/" --weights "/path/to/your/file.h5"```
* All these arguments are mandatory. Regarding the weights choice ```-w ```, You either can provide:
  * the path to a file ```weights.h5```
  * or enter ```coco``` or ```imagenet``` to make automaticaly downloading

To lauch detection:
* ```python main.py --command detection --origin_folder "/path/to/kaggle/data/folder/" --project_folder "/path/to/your/project/folder/" --weights "/path/to/your/file.h5"```
* All these arguments are mandatory.


## Exemple

![Image of detection](https://raw.githubusercontent.com/j-bd/mask_rcnn/master/detec.png)


## Citation

```@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```
