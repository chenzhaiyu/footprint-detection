#!/usr/bin/env python

import cv2
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import glob


def pixelacc(result, lable):
    data_diff = result - lable
    # data_diff[data_diff == 0] = 1
    # data_diff[data_diff != 1] = 0
    # pix_acc = sum(sum(data_diff))/512/512
    pix_acc = np.sum(data_diff == 0) / 512 / 512
    return pix_acc


# 1、计算检测结果的重叠区域并集：两张图像相加，非0的区域
# 2、计算正确检测区域：result图把所有0变为1，相减，值为0的区域

def IOU(result, lable):
    img_sum = result + lable
    # img_sum[img_sum != 0] = 1
    result[result == 0] = 2
    dif = result - lable
    # Iou_acc = np.sum(dif == 0) / np.sum(img_sum != 0)
    return [np.sum(dif == 0), np.sum(img_sum != 0)]


# 检测率：正确的房屋像元/模板
# 漏检率：
# 误检率：检测错误的像元/检测总像元
def checkacc(result, lable):
    lable[lable == 255] = 1
    dif = result - lable
    # jc_acc = np.sum(dif == 254) / np.sum(lable == 1)
    # lj_acc = 1 - jc_acc
    # wj_acc = np.sum(dif == 255) / np.sum(result == 255)
    return [np.sum(dif == 254), np.sum(lable == 1), np.sum(dif == 255), np.sum(result == 255)]


def evaluate():
    lable_path = "/Users/czy/Dataset/WHU Building Dataset/data/test/sub_label/"
    result_path = "/Users/czy/Documents/Projects/Pycharm/MaskRCNN/images/matting_mask/binary/"
    imgsname = glob.glob(lable_path + "*.tif")
    totall_pix_acc = 0
    Iou = []
    jc = []
    threshold = 0.5
    n = 0
    for imgname in imgsname:
        name = imgname[imgname.rindex("/") + 1:]
        img = load_img(result_path + name, color_mode="grayscale")
        img_gt = load_img(imgname, color_mode="grayscale")
        imgdata = img_to_array(img).astype(np.float32)
        imgdata_gt = img_to_array(img_gt).astype(np.float32)
        imgdata[imgdata > 255 * threshold] = 255
        imgdata[imgdata <= 255 * threshold] = 0

        # saveimg = array_to_img(imgdata)
        # saveimg.save("F:\\python_progect\\unet\\img_result\\"+name)

        pix_acc = pixelacc(imgdata, imgdata_gt)
        totall_pix_acc += pix_acc
        Iou.append(IOU(imgdata, imgdata_gt))
        jc.append(checkacc(imgdata, imgdata_gt))
        n += 1
        if n % 10 == 0:
            print(n, "张图像计算完毕！")
    Iou2 = sum(np.array(Iou))
    Iou_acc = Iou2[0]/Iou2[1]
    jc2 = sum(np.array(jc))
    jc_acc = jc2[0]/jc2[1]
    wj_acc = jc2[2]/jc2[3]
    totall_pix_acc = totall_pix_acc / len(imgsname)
    print("Iou: ", Iou_acc)
    print("recall: ", jc_acc)
    print("precision: ", 1 - wj_acc)
    print("像素精度: ", totall_pix_acc)


def evaluate_pj():
    lable_path = "F:\\python_progect\\unet\\resample2\\test\\label\\"
    result_path = "F:\\python_progect\\unet\\results\\"
    imgsname = glob.glob(lable_path + "*.tif")
    totall_pix_acc = 0
    Iou = []
    jc = []
    threshold = 0.55
    n = 0
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:]
        img = load_img(result_path + name, grayscale=True)
        img_gt = load_img(imgname, grayscale=True)
        label_sub2 = cv2.resize(img_gt, (256, 256), interpolation=cv2.INTER_AREA)
        label_pin_sub2 = np.zeros((512, 512), img_gt.dtype)
        label_pin_sub2[0:256, 0:256] = label_sub2
        label_pin_sub2[0:256, 256:512] = label_sub2
        label_pin_sub2[256:512, 0:256] = label_sub2
        label_pin_sub2[256:512, 256:512] = label_sub2

        imgdata = img_to_array(img).astype(np.float32)
        imgdata_gt = img_to_array(label_pin_sub2).astype(np.float32)
        imgdata[imgdata > 255 * threshold] = 255
        imgdata[imgdata <= 255 * threshold] = 0
        pix_acc = pixelacc(imgdata, imgdata_gt)
        totall_pix_acc += pix_acc
        Iou.append(IOU(imgdata, imgdata_gt))
        jc.append(checkacc(imgdata, imgdata_gt))
        n += 1
        if n % 200 == 0:
            print(n, "张图像计算完毕！")
    Iou2 = sum(np.array(Iou))
    Iou_acc = Iou2[0]/Iou2[1]
    jc2 = sum(np.array(jc))
    jc_acc = jc2[0]/jc2[1]
    wj_acc = jc2[2]/jc2[3]
    totall_pix_acc = totall_pix_acc / len(imgsname)
    print("Iou 精度: ", Iou_acc)
    print("检测率 recall: ", jc_acc)
    print("1-误检率=precision: ", 1 - wj_acc)
    print("像素精度: ", totall_pix_acc)

def evaluate_single():
    # lable_path = "E:\\ArcGIS\\resample2_512\\train\\lable\\"
    # lable_path = "E:\\ArcGIS\\building\\2\\label\\label\\"
    lable_path = "E:\\AerialImageDataset\\train\\all\\vienna\\label\\"
    result_path = "E:\\TensorFlow\\unet\\results\\"
    # resultimg_path = "E:\\TensorFlow\\unet\\resultimg\\"
    imgsname = glob.glob(lable_path + "*.tif")
    all_acc = []
    threshold = 0.25
    f = open("single_acc.txt", "w")
    f.write("文件名,Iou,检测率Recall,1-误检率precision,像素精度\n")
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:]
        img = load_img(result_path + name, grayscale=True)
        img_gt = load_img(imgname, grayscale=True)
        imgdata = img_to_array(img).astype(np.float32)
        imgdata_gt = img_to_array(img_gt).astype(np.float32)
        imgdata[imgdata > 255 * threshold] = 255
        imgdata[imgdata <= 255 * threshold] = 0

        pix_acc = pixelacc(imgdata, imgdata_gt)
        iou_out = IOU(imgdata, imgdata_gt)
        iou_acc = iou_out[0]/iou_out[1]
        check_out = checkacc(imgdata, imgdata_gt)
        if check_out[1] == 0:
            jc_acc = 0
            wj_acc = 1
        else:
            jc_acc = check_out[0]/check_out[1]
            wj_acc = check_out[2] / check_out[3]
        lj_acc = 1 - jc_acc
        if np.sum(imgdata_gt == 0) != 512*512:
            f.write(name + "," + str(iou_acc) + "," + str(jc_acc) + "," + str(1-wj_acc) + "," + str(pix_acc) + "\n")
        # all_acc.append(acc)
        # acc_np = np.array(all_acc)
    f.close()
    # print(acc_np)


def binarize(image_dir, output_dir):

      file_names = next(os.walk(image_dir))[2]

      image_count = 0

      for file_name in file_names:

            image_count = image_count + 1

            filenames_already = next(os.walk(output_dir))[2]
            if file_name in filenames_already:
                  print("already found " + file_name + " in directory")
                  continue

            print("------------processing " + file_name + " count: " + str(image_count) + str(" / ") + str(len(file_names)) +
                  " ------------\n")

            # Load a random image from the images folder
            image = cv2.imread(os.path.join(image_dir, file_name), cv2.IMREAD_COLOR)

            # binarize image
            ret, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

            cv2.imwrite(output_dir + file_name, binary)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    evaluate()