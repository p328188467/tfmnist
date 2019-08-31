# -*- coding:utf-8 -*-
import redis
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
TASKNAME='exnet13'
DATAROOT='/mnt/data1/'+TASKNAME
UNINIT=0
INITED=1
PLANTING=1
GROWING=2
HARVESTING=3
WORKER=206
TRAIN_EPOCH=3
CHECK_TIME=600

def total_secends(td):
  return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6

def train(model,msc,epoch):
    # Load training and eval data
    print(keras.__version__)
    datagen = ImageDataGenerator(
    ##  featurewise_center=True,
    ##  featurewise_std_normalization=True,
      rotation_range=20,
      width_shift_range=0.15,
      height_shift_range=0.15)
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
    
    remaintime = CHECK_TIME
    check_gap = 0.01/epoch
    if msc.lastacc == 0:	
      model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy'])

      batches = 0
      ##model.fit(train_data, train_labels, epochs=1, batch_size=64, verbose=1, validation_split=0.05)
      ##model.summary()    
      for  x_batch, y_batch in datagen.flow(train_data, train_labels, batch_size=256):
        model.fit(x_batch, y_batch,batch_size=32)
        batches += 1
    #   break
        if batches >= np.size(train_data,0) / 256:
          # 我们需要手动打破循环，
          # 因为生成器会无限循环
          break
      loss, accuracy = model.evaluate(eval_data, eval_labels)
      msc.last_acc = accuracy    
    
    while True:
      remaintime = CHECK_TIME
      last_acc = accuracy
      while remaintime > 0:
        begintime = datetime.datetime.now()
        batches = 0
        ##model.fit(train_data, train_labels, epochs=1, batch_size=64, verbose=1, validation_split=0.05)
        ##model.summary()    
        for  x_batch, y_batch in datagen.flow(train_data, train_labels, batch_size=256):
          model.fit(x_batch, y_batch,batch_size=32)
          batches += 1
    #      break
          if batches >= np.size(train_data,0) / 256:
            # 我们需要手动打破循环，
            # 因为生成器会无限循环
            break
        endtime = datetime.datetime.now()
        deltatime = (endtime-begintime)
        td = total_secends(deltatime)
        loss, accuracy = model.evaluate(eval_data, eval_labels)
        theta = deltaacc / td / (1-accuracy)
        remaintime = remaintime- td
        numpara = model.count_params()
        npfitness = 0.0001 * numpara/(28*28*10)
        msc.totaltime = msc.totaltime + td     
      deltaacc = accuracy - last_acc
      if deltaacc < check_gap:
        break
    
    print('Test loss:', loss)
    print('Accuracy:', accuracy)
    print('fitness:',theta)
    msc.last_acc = accuracy
    return loss,accuracy,npfitness

  
  
  
def main(unused_argv):
  workerID = WORKER
  pool = redis.ConnectionPool(host=REDISHOST, port=6379)
  r = redis.Redis(connection_pool=pool)
  
  dlm = redlock.Redlock([{"host": REDISHOST, "port": 6379, "db": 0}, ])
  
  while True:
    time.sleep(1)
    ret = r.hmget(TASKNAME,'epoch','status')
    t=task(TASKNAME,int(ret[0]),int(ret[1]))
    if t.status == GROWING:
      epochpath=DATAROOT+'/'+str(t.epoch)
      dirs = os.listdir(epochpath)
      for d in dirs:
        modelpath = os.path.join(epochpath, d)
        if os.path.isdir(modelpath):
          mod_lock = dlm.lock(modelpath,7200*1000)
          if not mod_lock:
            continue
          if os.path.exists(modelpath+'/'+'result'):
            continue
          modelfilepath = modelpath+'/'+d
          print('training modelpath:',modelfilepath)
          msc = ModelString.load(modelfilepath)
          mod = msc.load_model()
          msc.load_weights(mod,by_name=True)
          loss,acc,np =train(mod,msc,t.epoch+1)
          ##calculate fitness
          ##fitness 
          ret = Modresult(acc,acc-np,modelpath)
          
          msc.save_weights(mod)
          msc.save()
          ret.save()
          dlm.unlock(mod_lock)


  
if __name__ == "__main__":
  main(None)
    

