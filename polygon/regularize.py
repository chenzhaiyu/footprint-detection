import shapefile
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [10, 10]
matplotlib.rcParams['figure.subplot.left'] = 0
matplotlib.rcParams['figure.subplot.bottom'] = 0
matplotlib.rcParams['figure.subplot.right'] = 1
matplotlib.rcParams['figure.subplot.top'] = 1

import numpy as np
import os
import cv2
from skimage import measure


def regularize(image_grey, tolerance=2, image_save_path=None, shapefile_save_path=None, verbose=0):
    """
    regularize polygons using Douglas-Puck algorithm and save results as shapefile
    :param image_grey: grey image
    :param tolerance: tolerance parameter for Douglas-Peucker algorithm
    :param shapefile_save_path: shapefile save path
    :param image_save_path: image save path
    :param verbose: set 1 to print every contour or 0 to make it silent
    :return: return approximate_contours (list of arrays, with each dimension as n*2) if image_save_path
    and shapefile_save_path is None
    """
    assert len(image_grey.shape) == 2, "provided image dimension not supported!"
    assert not (image_save_path and shapefile_save_path), "either save result as image or shapefile, got both!"

    contours = measure.find_contours(image_grey, 100)

    if verbose and len(contours) > 0:
        print("length of contours: " + str(len(contours)))
        print("the last polygon: " + str(contours[-1]))

    if shapefile_save_path:

        w = shapefile.Writer(shapefile_save_path, shapefile.POLYGON)
        w.autoBalance = 1
        w.field('FIRST_FLD', 'C', '40')

        for n, contour in enumerate(contours):
            this_contour = contour.copy()
            approximate_contour = measure.approximate_polygon(this_contour, tolerance=tolerance)
            approximate_contour[:, [1, 0]] = approximate_contour[:, [0, 1]]
            approximate_contour[:, [1]] = 0 - approximate_contour[:, [1]]
            w.poly([approximate_contour.tolist()])
            w.record('')
        w.close()

    elif image_save_path:
        rows, cols = image_grey.shape
        plt.axis([0, rows, cols, 0])
        for n, contour in enumerate(contours):
            this_contour = contour.copy()
            approximate_contour = measure.approximate_polygon(this_contour, tolerance=tolerance)
            plt.plot(approximate_contour[:, 1], approximate_contour[:, 0], linewidth=2)
        plt.axis("off")
        plt.savefig(image_save_path)

    else:
        # No shapefile_save_path or image_save_path provided
        approximate_contours = []
        for n, contour in enumerate(contours):
            this_contour = contour.copy()
            approximate_contour = measure.approximate_polygon(this_contour, tolerance=tolerance)
            approximate_contours.append(approximate_contour)
        return approximate_contours


def display_shapefile(path_shapefile):
    """
    display shapefile
    :param path_shapefile: path to shapefile
    :return: no return
    """

    sf = shapefile.Reader(path_shapefile)

    plt.figure()
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y)
    plt.show()


def mask_to_contour(source_dir, suffix, save_dir):
    """
    turn mask image to contour image

    source_dir：source image dir
    suffix: such as "tif", "jpg"
    save_dir: result save dir
    """

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for file in os.listdir(source_dir):
        file_name = os.path.join(source_dir, file)
        file_form = os.path.splitext(file_name)[1][1:]
        if file_form == suffix:
            print('loading ', file_name)
            image = cv2.imread(file_name, 0)
            edge = np.zeros((image.shape[0], image.shape[1]))
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    if (image[i, j] == 255 and image[i + 1, j] == 0) or (image[i, j] == 255 and image[i - 1, j] == 0):
                        edge[i, j] = 255
                    if (image[i, j] == 255 and image[i, j + 1] == 0) or (image[i, j] == 255 and image[i, j - 1] == 0):
                        edge[i, j] = 255
            cv2.imwrite(os.path.join(save_dir, file), edge)


if __name__ == '__main__':

    image = cv2.imread("/Users/czy/Desktop/test/2_8.png", 0)
    regularize(image, shapefile_save_path=None, image_save_path="/Users/czy/Desktop/result.png")
