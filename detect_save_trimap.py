"""

Detect with Mask-RCNN, then save trimaps as files

"""

# TODO: combine detect_save_mask and detect_save_trimap as one .py file

import os
import sys
import numpy as np
import skimage.io

import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
PRE_MODEL_PATH = os.path.join(ROOT_DIR, "models/16_mAP_0.8625_pre_0.8761_rec_0.8856_60.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(PRE_MODEL_PATH):
    utils.download_trained_weights(PRE_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = "/Users/czy/Dataset/WHU Building Dataset/data/test/sub_image/"

# Image patch save path
PATCH_SAVE_DIR = "images/patches/"

# Trimap prediction save path
TRIMAP_SAVE_DIR = "images/trimaps/"


class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "BUILDING"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2


config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(PRE_MODEL_PATH, by_name=True)

class_names = ['BG', 'BUILDING']

file_names = next(os.walk(IMAGE_DIR))[2]

# set counter and bug counter
image_count = 0
bug_count = 0
output_count = 0
kernel = np.ones((7, 7), np.uint8)

for file_name in file_names:

    image_count = image_count + 1

    # skip this image if found in MASK_SAVE_DIR
    pred_mask_names = next(os.walk(TRIMAP_SAVE_DIR))[2]
    if file_name in pred_mask_names:
        print("already found " + file_name + " in directory")
        continue

    print("------------processing " + file_name + " count: " + str(image_count) + str(" / ") + str(len(file_names)) +
          " ------------\n")

    # Load a random image from the images folder
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

    # Run detection
    results = model.detect([image], verbose=0)

    # Visualize results
    r = results[0]

    # number of boxes
    N = results[0]['rois'].shape[0]
    # masks with 90% or higher confidence
    masks = results[0]['masks']
    # boxes
    boxes = results[0]['rois']

    if masks is None or boxes is None:
        continue

    for i in range(0, N):

        y1, x1, y2, x2 = boxes[i]

        patch = image[y1:y2, x1:x2]

        output_count = output_count + 1

        # original patch mask
        patch_mask_ori = 255 * masks[y1:y2, x1:x2, i].astype("uint8")

        # dilate the mask
        patch_mask_dilated = cv2.dilate(patch_mask_ori, kernel, iterations=1)

        # paint the mask as trimap, add halved original to halved dilated
        patch_mask_trimap = np.where((patch_mask_ori == 255), 127, 0) + np.where((patch_mask_dilated == 255), 128, 0)

        # TODO: save images using opencv
        # # save patch
        # plt.imsave(PATCH_SAVE_DIR + str(output_count) + ".png", patch, cmap=plt.cm.gray)
        #
        # # save trimap
        # plt.imsave(TRIMAP_SAVE_DIR + str(output_count) + ".png", patch_mask_trimap, cmap=plt.cm.gray)

        # gray_patch_mask_trimap = cv2.cvtColor(patch_mask_trimap, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(PATCH_SAVE_DIR + str(output_count) + ".png", patch)

        cv2.imwrite(TRIMAP_SAVE_DIR + str(output_count) + ".png", patch_mask_trimap)

print("------------All Done!------------")
print("Bug Count: " + str(bug_count))

