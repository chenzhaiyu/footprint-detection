"""

use Mask-RCNN to get patches, then use KNN matting to refine them

"""

from matting import knn_matting
from skimage import morphology

import scipy

import os
import sys
import numpy as np

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

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
PRE_MODEL_PATH = os.path.join(ROOT_DIR, "models/16_mAP_0.8625_pre_0.8761_rec_0.8856_60.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(PRE_MODEL_PATH):
    utils.download_trained_weights(PRE_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = "/Users/czy/Dataset/WHU Building Dataset/data/test/sub_image/"

# Trimap prediction save path
OUTPUT_DIR = "images/matting/"


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

output_count = 0
kernel = np.ones((7, 7), np.uint8)

for file_name in file_names:

    image_count = image_count + 1

    # skip this image if found in MASK_SAVE_DIR
    pred_mask_names = next(os.walk(OUTPUT_DIR))[2]
    if file_name in pred_mask_names:
        print("already found " + file_name + " in directory")
        continue

    print("------------processing " + file_name + " count: " + str(image_count) + str(" / ") + str(len(file_names)) +
          " ------------\n")

    # Load a random image from the images folder
    image = scipy.misc.imread(os.path.join(IMAGE_DIR, file_name))[:,:,:3]

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

    canvas = np.ones(image.shape[:2])
    black = np.zeros(image.shape[:2])

    for i in range(0, N):

        rcnnmask = masks[:, :, i]

        if not True in rcnnmask:
            cv2.imwrite(OUTPUT_DIR + file_name, black)
            continue

        y1, x1, y2, x2 = boxes[i]

        # patch = image[y1:y2, x1:x2] / 255.0

        output_count = output_count + 1

        # skeleton = morphology.skeletonize(rcnnmask)

        trimap = np.zeros(image.shape, np.uint8)  # initialize all sure background

        trimap[y1:y2, x1:x2, :3] = 128

        trimap[rcnnmask == True, :] = 255

        # run closed form matting
        alpha = knn_matting.knn_matte(image, trimap)

        # inverse the alpha for iterative painting
        canvas = canvas * (1 - alpha)

        # cv2.imwrite(OUTPUT_DIR + str(output_count) + "_image.png", image)
        # cv2.imwrite(OUTPUT_DIR + str(output_count) + "_trimap.png", trimap)
        # cv2.imwrite(OUTPUT_DIR + str(output_count) + "_alpha.png", alpha * 255.0)

    cv2.imwrite(OUTPUT_DIR + file_name, (1 - canvas) * 255.0)

print("------------All Done!------------")
