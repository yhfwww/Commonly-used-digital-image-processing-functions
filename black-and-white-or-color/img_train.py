# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:13:26 2017
卷积神经网络黑白彩色图像区分
@author: yaohongfu
"""
from __future__ import print_function
import random

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
import cv2

from img_import import extract_data,resize_with_pad,IMAGE_SIZE,read_image

class Dataset(object):
    def __init__(self):
        self.X_train=None
        self.X_valid=None
        self.X_test=None
        self.Y_train=None
        self.Y_valid=None
        self.Y_test=None
    def read(self,datapath,img_rows=IMAGE_SIZE,img_cols=IMAGE_SIZE,img_channels=3,nb_classes=2):
        images,labels=extract_data(datapath)
        labels=np.reshape(labels,[-1])
        X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=0.3,random_state=random.randint(0,100))
        X_valid,X_test,y_valid,y_test=train_test_split(images,labels,test_size=0.5,random_state=random.randint(0,100))
        if K.image_dim_ordering()=='th':
            X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
            X_valid = X_valid.reshape(X_valid.shape[0], 3, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
            X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 3)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
            input_shape=(img_rows,img_cols,3)
        # the data, shuffled and split between train and test sets
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')
        print('IMAGE_SIZE',IMAGE_SIZE)
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train,nb_classes)
        Y_valid = np_utils.to_categorical(y_valid,nb_classes)
        Y_test = np_utils.to_categorical(y_test,nb_classes)

        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_valid /= 255
        X_test /= 255

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_valid = Y_valid
        self.Y_test = Y_test
        

class Model(object):
    FILE_PATH=r'model1212.h5'
    def __init__(self):
        self.model = None
    def build_model(self,dataset,nb_classes=2):
        self.model=Sequential()
        #Convolution2D卷积核的数目32卷积大小3x3,border_mode边界模式same输入
        self.model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
        #input_shape=dataset.X_train.shape[1:])
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        
        self.model.add(Convolution2D(32,3,3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        
        #self.model.add(Dropout(0.5))
        self.model.add(Convolution2D(64,3,3,border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(800))#512
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))
        #self.model.summary()
    def train(self,dataset,batch_sizes=32,nb_epochs=40,data_augmentation=True):
        # let's train the model using SGD + momentum.
        sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
        self.model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(dataset.X_train,dataset.Y_train,batch_size=batch_sizes,nb_epoch=nb_epochs,validation_data=(dataset.X_valid,dataset.Y_valid),shuffle=True)
        else:
            print('使用实时数据增强')
            # this will do preprocessing and realtime data augmentation
            datagen=ImageDataGenerator(
                featurewise_center=False,             # set input mean to 0 over the dataset
                samplewise_center=False,              # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,   # divide each input by its std
                zca_whitening=False,                  # apply ZCA whitening
                rotation_range=20,                    # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,                # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,               # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,                 # randomly flip images
                vertical_flip=False)              # randomly flip images
            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(dataset.X_train)
            # fit the model on the batches generated by datagen.flow()
            self.model.fit_generator(datagen.flow(dataset.X_train,dataset.Y_train,batch_size=batch_sizes),samples_per_epoch=dataset.X_train.shape[0],nb_epoch=nb_epochs,validation_data=(dataset.X_valid,dataset.Y_valid))
    def save(self,file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self,file_path=FILE_PATH):
        print('Model Loaded.')
        self.model=load_model(file_path)

    def predict(self,image):
        if K.image_dim_ordering()=='th' and image.shape !=(1,3,IMAGE_SIZE,IMAGE_SIZE):
            image=resize_with_pad(image)
            image=image.reshape((1,3,IMAGE_SIZE,IMAGE_SIZE))
        elif K.image_dim_ordering()=='tf'and image.shape!=(1,IMAGE_SIZE,IMAGE_SIZE,3):
            image=resize_with_pad(image)#尺寸
            image = image.reshape((1,IMAGE_SIZE,IMAGE_SIZE,3))
        image = image.astype('float32')
        image/=255
        result=self.model.predict_proba(image)
        print(result)
        result = self.model.predict_classes(image)
        return result[0]
    def predict_who(self,img):
        if K.image_dim_ordering()=='th' and img.shape !=(1,3,IMAGE_SIZE,IMAGE_SIZE):
            img=resize_with_pad(img)
            img=img.reshape((1,3,IMAGE_SIZE,IMAGE_SIZE))
        elif K.image_dim_ordering()=='tf' and img.shape !=(1,IMAGE_SIZE,IMAGE_SIZE,3):
            img=resize_with_pad(img)
            img=img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
        img=img.astype('float32')
        result1=self.model.predict_proba(img)
        print(result1)
        result2=self.model.predict_classes(img)
        if result2[0]==0:
            who='black-and-white pictures'
        else:
            who='color photograph'
        return who
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))


path=r'..\img'
#加载数据
setimg=Dataset()
setimg.read(path)


model=Model()
model.build_model(setimg)
model.train(setimg,nb_epochs=50,data_augmentation=False)
model.evaluate(setimg)

#单图像运用
img=cv2.imread(r'..\img\1\19.jpg')
pre=model.predict_who(img)
font=cv2.FONT_HERSHEY_SIMPLEX
img=cv2.putText(img,pre,(100,100),font,1.5,(255,0,0),3)
#添加文字，1.5表示字体大小，（100,100）是初始的位置，(255,0,0)表示颜色，3表示粗细
cv2.imshow("img",img)  
cv2.waitKey(0)  
cv2.destroyAllWindows() 


#文件夹批量运用
import os
testpath=r'..\img\1'
def predict_test(path):
    resultlist=[]
    for file_or_dir in os.listdir(path):
        abs_path=os.path.abspath(os.path.join(path,file_or_dir))
        if os.path.isdir(abs_path):
            predict_test(abs_path)
        else:
            if file_or_dir.endswith('.jpg'):
                img=cv2.imread(abs_path)
                result=model.predict_who(img)
                print(abs_path)
                print(result)
                resultlist.append(result)
    return resultlist


predict=predict_test(testpath)