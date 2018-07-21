# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 21:07:22 2018

@author: Narayanan Abishek
"""
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import tensorflow as tf

import numpy as np

from keras.backend import tensorflow_backend as K
with tf.Session(config = tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
    
    K.set_session(sess)
    if K =='tensorflow':
        K.set_image_dim_ordering("th")

 
    from keras.datasets import cifar10

    batch_size = 32
    num_classes = 10
    epochs = 15

    (x_train, y_train), (x_val, y_val) = cifar10.load_data() 

    num_classes = len(np.unique(y_train))

    class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

    #normalizing rgb values by dividing by 255.
    x_train = x_train.astype('float32')/255
    x_val = x_val.astype('float32')/255
    
    #sets output labels as one-hot encoded vectors such as 1 in the position of the class and 0 in rest of all the positions of the vector
    
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_val = np_utils.to_categorical(y_val, num_classes)
    
    #printing size of train and test data for verification
    print("\nThe train data features' size is:", x_train.shape)
    print("\nThe train data labels' size is:",y_train.shape)
    print("\nThe test data features'size is:",x_val.shape)
    print("\nThe test data labels' size is:",y_val.shape)
    
    def modelcnn():
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), padding='same', input_shape=(3, 32, 32)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(32, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        
        return model
    
    train_model = modelcnn()
    trained_model = train_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val,y_val),shuffle=True)
    scores = train_model.evaluate(x_val, y_val, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    plt.figure(0)
    plt.plot(trained_model.history['acc'],'r')
    plt.plot(trained_model.history['val_acc'],'g')
    plt.xticks(np.arange(0, 16, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['Train_data','Validation_data'])
    
    
    plt.figure(1)
    plt.plot(trained_model.history['loss'],'r')
    plt.plot(trained_model.history['val_loss'],'g')
    plt.xticks(np.arange(0, 16, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['Train_data','Validation_data'])
    plt.show()