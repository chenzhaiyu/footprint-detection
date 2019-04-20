import numpy as np
from skimage import morphology
import cv2


def grabcut(image, mask, x1=None, x2=None, y1=None, y2=None, mode="mask"):

    # Initialize _mask as sure background for GrabCut
    _mask = np.zeros(image.shape[:2], np.uint8)

    if mode == "skeleton":
        mask = morphology.skeletonize(mask.astype("uint8"))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if x1 is None:
        _mask[:, :] = cv2.GC_PR_BGD
    else:
        _mask[y1:y2, x1:x2] = cv2.GC_PR_BGD

    _mask[mask == True] = cv2.GC_FGD

    # GrabCut often gets assertion failure when image or mask size is weird
    try:
        _mask, bgd_model, fgd_model = cv2.grabCut(image, _mask.astype("uint8"), None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)

    except:
        print("Bug Occurs with cv2.grabCut!")
        pass

    # Covert _mask to human known format
    # Change sure background and probable background to 0
    _mask = 255 * np.where((_mask == 2) | (_mask == 0), 0, 1).astype('uint8')

    return _mask



