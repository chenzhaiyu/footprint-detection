import shapefile
import matplotlib as mpl
mpl.use('TkAgg')

import cv2
import matplotlib.pyplot as plt
from skimage import measure


def regularize(image_grey):
    w = shapefile.Writer(shapefile.POLYGON)
    w.autoBalance = 1
    w.field('FIRST_FLD', 'C', '40')
    # img_contour = image_grey.copy()
    # img_contour[img_contour > 0] = 0
    contours = measure.find_contours(image_grey, 100)
    print(len(contours))
    if len(contours) > 0:
        print(contours[-1])
    for n, contour in enumerate(contours):
        new_img = contour.copy()
        appr_img = measure.approximate_polygon(new_img, tolerance=2)
        appr_img[:, [1, 0]] = appr_img[:, [0, 1]]
        appr_img[:, [1]] = 0 - appr_img[:, [1]]
        w.poly(parts=[appr_img.tolist()])
        w.record('')

    w.save('shp_result.shp')


def display_shape(path_shapefile):

    sf = shapefile.Reader(path_shapefile)

    plt.figure()
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y)
    plt.show()


if __name__ == '__main__':

    img = cv2.imread("detection.png", cv2.IMREAD_GRAYSCALE)
    regularize(img)

    display_shape("shp_result.shp")