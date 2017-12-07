# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:52:23 2017

@author: yaohongfu
"""

#%matplotlib inline
import time
import os
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
def get_time_str():
    #获取当前字符型时间
    return(time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime((time.time()))))
def search(path,word):
    #查找某文件夹下文件名包含某关键字的文件列表
    result_file=[]
    for filename in os.listdir(path):
        fp = os.path.join(path,filename)
        if os.path.isfile(fp) and word in filename:
            result_file.append(fp)
        else:
            continue
    return result_file
def imgs_subplot(img_list):
    #img_list要显示的图片列表,列表里面是图片数组
    num=len(img_list)
    hang=int((num-0.1)/3.0+1)
    lie=3
    fig=plt.figure()
    for i in range(num):
        ax=fig.add_subplot(hang,lie,i+1)
        ax.imshow(img_list[i]) 
    plt.show()
def gamma_trans(img,gamma):
    #定义Gamma矫正的函数
    #采用Gamma校正法对输入图像进行颜色空间的标准化（归一化），目的是调节图像的对比度，
    #降低图像局部的阴影和光照变化所造成的影响，同时可以抑制噪音。一般采用的gamma值为0.5。
    #一般使用灰度图像，也可以是rgb图像
    #具体做法是先归一化到1,然后gamma作为指数值求出新的像素值再还原
    #返回处理后的图像
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    #实现这个映射用的是OpenCV的查表函数
    return cv2.LUT(img,gamma_table)
def get_face(img_file,name,size):
    #头像截取，单图处理函数
    #img_file图片路径
    #name新名称含格式
    #size新尺寸
    face_patterns = cv2.CascadeClassifier(r'C:\Downloads\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml')
    sample_image= cv2.imread(img_file)
    faces=face_patterns.detectMultiScale(sample_image,scaleFactor=1.2,minNeighbors=5,minSize=(30,30))
    for (x,y,w,h) in faces:
        face_of_img=sample_image[(y):(y+w),(x):(x+h),:]
        #print(type(face_of_img))
        #image=transform.resize(face_of_img,(size,size))#改变尺寸
        res=cv2.resize(face_of_img,(size,size),interpolation=cv2.INTER_AREA)
        #CV_INTER_NN-最近邻插值,CV_INTER_LINEAR-双线性插值(缺省使用)
        #CV_INTER_AREA-使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN 方法
        #CV_INTER_CUBIC - 立方插值.  
        cv2.imwrite(str(x)+name,res)
def get_fece_by_dir(ori_dir,to_dir,size):
    #批量人脸图像切割
    #原始文件夹到人脸存放文件夹
    for i in os.listdir(ori_dir):
        abs_ori_pic=os.path.join(ori_dir,i)
        abs_to_pic=os.path.join(to_dir,i)
        get_face(abs_ori_pic,abs_to_pic,size)
        
def re_ImageDataGenerator(path_a,path_b,num):
    #图片增生器，生成变换后的图像
    #path_a原路径path_b存放路径，num每一张图像生成的数量
    datagen=ImageDataGenerator(rotation_range=15,width_shift_range=0.15,height_shift_range=0.15,shear_range=0.150,zoom_range=0.10,horizontal_flip=False,fill_mode='nearest')
    for imgfile in os.listdir(path_a):
        abs_img=os.path.join(path_a,imgfile)
        img=load_img(abs_img)
        x=img_to_array(img)
        x=x.reshape((1,) + x.shape)
        i=0
        save_prefix_now=imgfile.split('_0')[0]
        for batch in datagen.flow(x,batch_size=1,save_to_dir=path_b,save_prefix=save_prefix_now,save_format='jpg'):
            i+=1
            if i >=num:
                break
                
def img_base64encode(img_path):
    #输入为图片路径
    #输出为图片的编码
    f=open(img_path,'rb')#二进制方式打开图文件
    ls_f=base64.b64encode(f.read())#读取文件内容，转换为base64编码
    f.close()
    return ls_f
def img_base64decode(img_64code):
    #输入为64编码
    #保存为图片
    imgData = base64.b64decode(img_64code)
    ly=open(r'temp.jpg','wb')
    ly.write(imgData)
    ly.close()
    img=cv2.imread(r'temp.jpg')
    return img

def scan_files(directory,prefix=None,postfix=None):
    #扫描文件夹下的所有图像或文件，返回图片路径列表
    #postfix可以指定后缀
    #prefix指定文件名前缀
    files_list=[]    
    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(root,special_file))
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(os.path.join(root,special_file))
            else:
                files_list.append(os.path.join(root,special_file))
    return files_list

def cut_pad(imgrgb):
    #图像去白边
    #参数数bgr图像数组
    img = cv2.cvtColor(imgrgb,cv2.COLOR_BGR2GRAY)
    h,w=img.shape
    top=0
    bot=0
    left=0
    right=0
    for i in range(h):
        if i<0.25*h and img[i,:].std()<0.7:
            top+=1
        if i >0.75*h and img[i,:].std()<0.7:
            bot+=1
    for j in range(w):
        if j <0.25*w and img[:,j].std()<0.7:
            left+=1
        if j >0.75*w and img[:,j].std()<0.7:
            right+=1
    return imgrgb[top:h-bot,left:w-right]