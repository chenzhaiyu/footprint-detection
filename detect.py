"""

Detection Module

"""

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

# Import Mask-RCNN
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

# Configs
USE_MATTING = True
SAVE_PATCH = False
SAVE_TRIMAP = False
BBOX_PADDING = False
matting_strategies = ["multi_shots", "one_shot"]
matting_methods = ["knn", "closed_form", "grabcut_skeleton", "grabcut_foreground"]

if USE_MATTING:
    matting_strategy = matting_strategies[0]
    matting_method = matting_methods[0]
    DILATION = True
    EROSION = True

else:
    matting_strategy = None
    matting_methods = None
    EROSION = None
    matting_method = None

# Directory of images to run detection on
IMAGE_SRC_DIR = "/Users/czy/Dataset/WHU Building Dataset/data/test/sub_image/"

if SAVE_PATCH:
    # Image patch save path
    PATCH_SAVE_DIR = "images/patches/"

else:
    PATCH_SAVE_DIR = None

if SAVE_TRIMAP:
    TRIMAP_SAVE_DIR = "images/trimaps/"

else:
    TRIMAP_SAVE_DIR = None

if BBOX_PADDING:
    MARGIN = 4

# Prediction result save path
OUTPUT_DIR = "images/trimaps/"


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

file_names = next(os.walk(IMAGE_SRC_DIR))[2]

# Set counter and bug counter
image_count = 0
output_count = 0

if USE_MATTING:
    dilation_kernel = np.ones((7, 7), np.uint8)
else:
    dilation_kernel = None

# Run predictions
for file_name in file_names:

    image_count = image_count + 1

    # Skip this image if found in MASK_SAVE_DIR
    output_mask_names = next(os.walk(OUTPUT_DIR))[2]
    if file_name in output_mask_names:
        print("already found " + file_name + " in directory")
        continue

    print("------------processing " + file_name + " count: " + str(image_count) + str(" / ") + str(len(file_names)) +
          " ------------\n")

    # Load a random image from the images folder
    image = skimage.io.imread(os.path.join(IMAGE_SRC_DIR, file_name))

    # Run detection
    results = model.detect([image], verbose=0)

    # Visualize results
    r = results[0]

    # Number of boxes
    N = results[0]['rois'].shape[0]

    # Masks with 90% or higher confidence
    # TODO: tune the confidence parameter
    masks = results[0]['masks']
    # probs = results[0]['probs']

    # Boxes
    boxes = results[0]['rois']

    if masks is None or boxes is None:
        UserWarning("No masks or boxes")
        continue

    if USE_MATTING and matting_strategy == "multi_shots":
        for i in range(0, N):

            rcnnmask = masks[:, :, i]

            y1, x1, y2, x2 = boxes[i]

            if BBOX_PADDING:
                if y1 >= MARGIN:
                    y1 = y1 - MARGIN
                if y2 < rcnnmask.shape[0] - MARGIN:
                    y2 = y2 + MARGIN
                if x1 >= MARGIN:
                    x1 = x1 - MARGIN
                if x2 < rcnnmask.shape[0] - MARGIN:
                    x2 = x2 + MARGIN

            patch = image[y1:y2, x1:x2]

            output_count = output_count + 1

            # patch mask
            patch_mask = 255 * masks[y1:y2, x1:x2, i].astype("uint8")

            # TODO: further erode the mask or not
            pass
            # dilate the mask to generate trimap
            dilation = cv2.dilate(patch_mask, dilation_kernel, iterations=1)

            # paint the mask as trimap, add halved original to halved dilated
            trimap = np.where((patch_mask == 255), 127, 0) + np.where((dilation == 255), 128, 0)

            if SAVE_PATCH:
                # save patches
                plt.imsave(PATCH_SAVE_DIR + str(output_count) + ".png", patch, cmap=plt.cm.gray)

            if SAVE_TRIMAP:
                # save trimap
                plt.imsave(TRIMAP_SAVE_DIR + str(output_count) + ".png", trimap, cmap=plt.cm.gray)

            # TODO: implement different mattings
            if matting_method == "closed_form":
                pass

            elif matting_method == "knn":
                pass

            elif matting_method == "grabcut_skeleton":
                pass

            elif matting_method == "grabcut_mask":
                pass

            else:
                UserWarning("Unknown matting method parameter")

    elif USE_MATTING and matting_strategy == "one_shot":
        # TODO: finish one_shot matting strategy
        pass

    else:
        # TODO: finish no-matting case
        pass

print("------------All Done!------------")

