#!/usr/bin/env python

import numpy as np
import cv2
import os
def face2line(dirFile,suffix,newDirFile):
    '''把dirFile文件夹下的类别图转化成边界图
    dirFile：指定的被转化的图片所在的文件夹
    suffix:图片类型，如“tif"
    newDirFile:转化出来的图片所要存的文件夹
    '''
    if(os.path.exists(newDirFile)==False):
        os.mkdir(newDirFile)
    for file in os.listdir(dirFile):
        singleFileName=os.path.join(dirFile, file)
        singlefileForm = os.path.splitext(singleFileName)[1][1:]
        if(singlefileForm == suffix):
            print('loading................ : ',singleFileName)
            img=cv2.imread(singleFileName,0)
            edge=np.zeros((img.shape[0],img.shape[1]))
            for i in range(1,img.shape[0]-1):
                for j in range(1,img.shape[1]-1):
                    if (img[i,j]==255 and img[i+1,j]==0) or(img[i,j]==255 and img[i-1,j]==0):
                        edge[i,j]=255
                    if (img[i,j]==255 and img[i,j+1]==0) or(img[i,j]==255 and img[i,j-1]==0):
                        edge[i,j]=255
            cv2.imwrite(os.path.join(newDirFile, file), edge)
dirFile="/Users/czy/Desktop"
suffix="png"
newDirFile="/Users/czy/Desktop/result"
face2line(dirFile,suffix,newDirFile)