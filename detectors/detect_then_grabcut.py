"""

Detect with Mask-RCNN, resulting skeletons, then use GrabCut to find the optimal segmentation

"""

import os
import sys
import numpy as np
import skimage.io

import cv2

from skimage import morphology

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

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

# Mask prediction save path
MASK_SAVE_DIR = "images/mask_predictions/WHU_test_12.21_grabcut/"


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
count = 0
bug_count = 0

for file_name in file_names:

    count = count + 1

    # skip this image if found in MASK_SAVE_DIR
    pred_mask_names = next(os.walk(MASK_SAVE_DIR))[2]
    if file_name in pred_mask_names:
        print("already found " + file_name + " in directory")
        continue

    print("------------processing " + file_name + " count: " + str(count) + str(" / ") + str(len(file_names)) +
          " ------------\n")

    # Load a random image from the images folder
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
    mask_image_save_path = MASK_SAVE_DIR + file_name

    # Run detection
    results = model.detect([image], verbose=0)

    # Visualize results
    r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'])

    # number of boxes
    N = results[0]['rois'].shape[0]
    masks = results[0]['masks']
    boxes = results[0]['rois']
    probs = results[0]['probs']  # mask probability

    if masks is None or boxes is None:
        continue

    canvas = np.ones(image.shape[:2])
    black = np.zeros(image.shape[:2])

    for i in range(0, N):

        rcnnmask = masks[:, :, i]
        maskprob = probs[:, :, i]

        if not True in rcnnmask:
            plt.imsave(mask_image_save_path, black, cmap=plt.cm.gray)
            continue

        # padding the bounding box, to ensure better foreground detection
        margin = 0
        y1, x1, y2, x2 = boxes[i]
        if y1 >= margin:
            y1 = y1 - margin
        if y2 < rcnnmask.shape[0] - margin:
            y2 = y2 + margin
        if x1 >= margin:
            x1 = x1 - margin
        if x2 < rcnnmask.shape[0] - margin:
            x2 = x2 + margin

        skeleton = morphology.skeletonize(rcnnmask)

        # specify image and model
        mask = np.zeros(image.shape[:2], np.uint8)  # initialize all sure background
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # specify rect coordinates for detection
        mask[y1:y2, x1:x2] = cv2.GC_PR_BGD  # rect area used as possible background

        # mask[rcnnmask == True] = cv2.GC_PR_FGD  # rcnn mask area used as possible foreground
        # mask[skeleton == True] = cv2.GC_FGD
        mask[rcnnmask == True] = cv2.GC_FGD

        # run GrabCut
        try:
            mask, bgdModel, fgdModel = cv2.grabCut(image, mask, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)

        except:
            print("Bug Occurs with" + str(file_name))
            bug_count = bug_count + 1
            mask = black

        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # TODO: change fuse type
        # function: turn mask to black
        mask = np.where((mask == 1), 0, 1).astype('uint8')

        # # function: turn mask to black, and union with rcnnmask
        # mask = np.where((mask | rcnnmask == 1), 0, 1).astype('uint8')

        # # function: turn mask to black, and intersect with rcnnmask
        # mask = np.where((mask & rcnnmask == 1), 0, 1).astype('uint8')

        canvas = canvas * mask

    # invert colors: paint mask with white (1)
    mask_image = np.where((canvas == 1), 0, 1).astype('bool')
    plt.imsave(mask_image_save_path, mask_image, cmap=plt.cm.gray)

print("------------All Done!------------")
print("Bug Count: " + str(bug_count))

