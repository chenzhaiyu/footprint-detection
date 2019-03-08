import cv2
import os

IMAGE_DIR = "/Users/czy/Documents/Projects/Pycharm/MaskRCNN/images/matting_mask/"
OUTPUT_DIR = "/Users/czy/Documents/Projects/Pycharm/MaskRCNN/images/matting_mask/binary/"

file_names = next(os.walk(IMAGE_DIR))[2]

image_count = 0

for file_name in file_names:

    image_count = image_count + 1

    filenames_already = next(os.walk(OUTPUT_DIR))[2]
    if file_name in filenames_already:
        print("already found " + file_name + " in directory")
        continue

    print("------------processing " + file_name + " count: " + str(image_count) + str(" / ") + str(len(file_names)) +
          " ------------\n")

    # Load a random image from the images folder
    image = cv2.imread(os.path.join(IMAGE_DIR, file_name), cv2.IMREAD_COLOR)

    # binarize image
    ret, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    cv2.imwrite(OUTPUT_DIR + file_name, binary)
