# -*- coding:utf-8 -*-
import pickle
import keras
import time
from keras.models import Model
from mnist_model import ExNet
from mnist_node import NetNode,ModelString, Modresult
from keras.layers import Input, add, Reshape, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.utils import plot_model
import os
import numpy as np

from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
from mnist_exmaster import task
import redlock
import math
import datetime
from keras.preprocessing.image import ImageDataGenerator 

DATA_DIR = './fashion-minst/'

REDISHOST='10.200.71.206'
#REDISHOST='127.0.0.1'
TASKNAME='exnet14'
DATAROOT='/mnt/data1/'+TASKNAME
UNINIT=0
INITED=1
PLANTING=1
GROWING=2
HARVESTING=3
WORKER=206
TRAIN_EPOCH=30
CHECK_TIME=600

def total_secends(td):
  return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6

def train(model,msc):
    # Load training and eval data
    print(keras.__version__)
    datagen = ImageDataGenerator(
    ##  featurewise_center=True,
    ##  featurewise_std_normalization=True,
      rotation_range=20,
      width_shift_range=0.10,
      height_shift_range=0.10)
    ##  horizontal_flip=True)
    
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True, validation_size=0)
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_data, train_labels = shuffle(train_data, train_labels)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)
    train_data = train_data.reshape(-1,28,28,1)  # Returns np.array
    eval_data = eval_data.reshape(-1,28,28,1)  # Returns np.array

    input_shape = (28, 28, 1)
    
    last_acc = msc.lastacc
    if msc.lastacc == 0:	
      model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.Adamax(),##adam
              metrics=['accuracy'])

    datagen.fit(x_train)
    best_weights_filepath = './best_weights.hdf5'
    earlyStopping = kcallbacks.EarlyStopping(monitor='accuracy', patience=5, verbose=1, mode='max')
    saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='auto')

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=TRAIN_EPOCH, workers=4, callbacks=[earlyStopping, saveBestModel, TensorBoard(log_dir='./tmp/log')])
    
    loss, accuracy = model.evaluate(eval_data, eval_labels)
    
    print('Test loss:', loss)
    print('Accuracy:', accuracy)

    return loss,accuracy,npfitness

  
  
  
def main(unused_argv):

  if len(argv) > 0:
    modelpath = argv[0]
    d = argv[1]
    modelpath = os.path.join(epochpath, d)
    modelfilepath = modelpath+'/'+d
    print('training modelpath:',modelfilepath)
    msc = ModelString.load(modelfilepath)
    mod = msc.load_model()
    msc.load_weights(mod,by_name=True)
    loss,acc,np =train(mod,msc)
    mod.summary()
    exit()



  
if __name__ == "__main__":
  main(None)
    

