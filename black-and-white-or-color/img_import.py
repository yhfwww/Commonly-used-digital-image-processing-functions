# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:11:01 2017
卷积神经网络，数据处理模块
@author: yaohongfu
"""

import os
import numpy as np
import cv2

IMAGE_SIZE=100#64

def resize_with_pad(image,height=IMAGE_SIZE,width=IMAGE_SIZE):
    #缩放到设置大小，短边方向补充黑块
    def get_padding_size(image):
        h,w,_=image.shape#获得尺寸大小
        longest_edge=max(h,w)#获得长边的尺寸
        top,bottom,left,right=(0,0,0,0)
        if h<longest_edge:#如果高不是长边
            dh=longest_edge-h
            top=dh/2
            bottom=dh-top
        elif w<longest_edge:
            dw=longest_edge-w
            left=dw/2
            right=dw-left
        else:
            pass
        return int(top),int(bottom),int(left),int(right)
    top,bottom,left,right=get_padding_size(image)
    BLACK=[0,0,0]
    constant=cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)
    resized_image=cv2.resize(constant,(height,width))
    return resized_image

def read_image(file_path):
    #读取图片并重塑大小
    image=cv2.imread(file_path)
    image=resize_with_pad(image,IMAGE_SIZE,IMAGE_SIZE)
    return image
images=[]
labels=[]

def traverse_dir(path):
    #变量文件夹读入图片数据和标签数据
    for file_or_dir in os.listdir(path):
        abs_path=os.path.abspath(os.path.join(path,file_or_dir))
        if os.path.isdir(abs_path):
            traverse_dir(abs_path)
        else:
            if file_or_dir.endswith('.jpg'):
                image=read_image(abs_path)
                images.append(image)
                lab=os.path.split(path)
                labels.append(lab[-1])
    return images,labels

def extract_data(path):
    #提取数据
    images,labels=traverse_dir(path)
    images=np.array(images)
    return images,labels