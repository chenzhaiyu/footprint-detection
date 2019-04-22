"""

Detection Module

"""

import os
import sys
import numpy as np
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

from matting.grabcut import grabcut
from matting.knn_matting import knn_matte
from matting.closed_form_matting import closed_form_matting_with_trimap, closed_form_matting_with_prior

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
SAVE_TRIMAP = True
BBOX_PADDING = False
DILATION = True
EROSION = True
# TODO: call levin's method for foreground / background reconstruction
USE_LEVIN_RECONSTRUCTION = False
matting_strategies = ["multi_shots", "one_shot"]
matting_methods = ["knn", "closed_form", "grabcut_skeleton", "grabcut_foreground"]
MARGIN = 4
ALPHA_TO_MASK_THRESHOLD = 0.8
dilation_kernel = np.ones((7, 7), np.uint8)

matting_strategy = matting_strategies[0]
matting_method = matting_methods[1]

# Directory of images to run detection on
IMAGE_SRC_DIR = "/Users/czy/Dataset/WHU Building Dataset/data/test/sub_images/"

# Image patch save dir
PATCH_SAVE_DIR = "images/patches/"

# Trimap save dir
TRIMAP_SAVE_DIR = "images/trimaps/"

# Prediction result save path
OUTPUT_DIR = "images/results/"


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

    # Load an image from the images folder
    # TODO: check if it reads ok
    image = cv2.imread(os.path.join(IMAGE_SRC_DIR, file_name))

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
        overlay_mask = np.zeros((masks.shape[:-1]), dtype="uint8")
        overlay_trimap = np.zeros((masks.shape[:-1]), dtype="uint8")
        for i in range(0, N):

            mask = masks[:, :, i]

            y1, x1, y2, x2 = boxes[i]

            if BBOX_PADDING:
                if y1 >= MARGIN:
                    y1 = y1 - MARGIN
                if y2 < mask.shape[0] - MARGIN:
                    y2 = y2 + MARGIN
                if x1 >= MARGIN:
                    x1 = x1 - MARGIN
                if x2 < mask.shape[0] - MARGIN:
                    x2 = x2 + MARGIN

            patch = image[y1:y2, x1:x2]

            output_count = output_count + 1

            # Patch mask (0 or 1)
            patch_mask = masks[y1:y2, x1:x2, i].astype("uint8")

            # TODO: further erode the mask or not
            pass
            # Dilate the mask to generate trimap
            patch_dilation = cv2.dilate(patch_mask, dilation_kernel, iterations=1)

            # Paint the mask as trimap, add halved original to halved dilated
            _patch_trimap = (np.where((patch_mask == 1), 127, 0) + np.where((patch_dilation == 1), 128, 0)).astype("uint8")

            # Turn 1-chanel to 3-chanel
            patch_trimap = cv2.cvtColor(_patch_trimap, cv2.COLOR_GRAY2BGR)

            if SAVE_PATCH:
                # Save patches
                plt.imsave(PATCH_SAVE_DIR + str(output_count) + ".png", patch, cmap=plt.cm.gray)

            # TODO: implement more mattings
            patch_result = None
            if matting_method == "closed_form":
                try:
                    _patch_result = closed_form_matting_with_trimap(patch, _patch_trimap)
                except:
                    # TODO: specify exception clause
                    print("closed_form matting encountered error!")
                    continue  # Skip this patch

                patch_result = (_patch_result >= ALPHA_TO_MASK_THRESHOLD).astype("uint8") * 255

            elif matting_method == "knn":
                _patch_result = knn_matte(patch, patch_trimap)
                if _patch_result is None:
                    continue
                patch_result = (_patch_result >= ALPHA_TO_MASK_THRESHOLD).astype("uint8") * 255

            elif matting_method == "grabcut_skeleton":
                patch_result = grabcut(patch, patch_mask, mode="skeleton")

            elif matting_method == "grabcut_mask":
                patch_result = grabcut(patch, patch_mask, mode="mask")

            else:
                UserWarning("Unknown matting method parameter")

            overlay_mask[y1:y2, x1:x2] = patch_result
            overlay_trimap[y1:y2, x1:x2] = _patch_trimap

        if SAVE_TRIMAP:
            # Save trimap
            cv2.imwrite(TRIMAP_SAVE_DIR + str(image_count) + ".png", overlay_trimap)

        cv2.imwrite(OUTPUT_DIR + str(image_count) + ".png", overlay_mask)

    elif USE_MATTING and matting_strategy == "one_shot":

        result = None
        overlay_mask = np.zeros((masks.shape[:-1]), dtype=bool)
        for i in range(masks.shape[-1]):
            overlay_mask += masks[:, :, i]

        overlay_mask = overlay_mask.astype("uint8")

        # Process the overlay mask
        # TODO: further erode the mask or not
        pass
        # dilate the mask to generate trimap
        dilation = cv2.dilate(overlay_mask, dilation_kernel, iterations=1)

        # paint the mask as trimap, add halved original to halved dilated
        _trimap = (np.where((overlay_mask == 1), 127, 0) + np.where((dilation == 1), 128, 0)).astype("uint8")

        # Turn 1-chanel to 3-chanel
        trimap = cv2.cvtColor(_trimap, cv2.COLOR_GRAY2BGR)

        if SAVE_TRIMAP:
            # Save trimap
            cv2.imwrite(TRIMAP_SAVE_DIR + str(image_count) + ".png", trimap)

        # TODO: implement more mattings
        if matting_method == "closed_form":
            _result = closed_form_matting_with_trimap(image, _trimap)
            result = (_result >= ALPHA_TO_MASK_THRESHOLD).astype("uint8") * 255

        elif matting_method == "knn":
            _result = knn_matte(image, trimap)
            result = (_result >= ALPHA_TO_MASK_THRESHOLD).astype("uint8") * 255

        elif matting_method == "grabcut_skeleton":
            result = grabcut(image, overlay_mask, mode="skeleton")

        elif matting_method == "grabcut_mask":
            result = grabcut(image, overlay_mask, mode="mask")

        else:
            UserWarning("Unknown matting method parameter")

        cv2.imwrite(OUTPUT_DIR + str(image_count) + ".png", result)

    else:

        overlay_mask = np.zeros((masks.shape[:-1]), dtype=bool)
        for i in range(masks.shape[-1]):
            overlay_mask += masks[:, :, i]
        cv2.imwrite(OUTPUT_DIR + str(image_count) + ".png", overlay_mask)


print("------------All Done!------------")

