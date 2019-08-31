# -*- coding:utf-8 -*-
import redis
import pickle
import keras
import time
from keras.models import Model
from mnist_model import ExNet
from mnist_node import NetNode, ModelString, Modresult
from keras.layers import Input, add, Reshape, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.utils import plot_model
import os, errno
import sys

#REDISHOST='10.200.71.206'
REDISHOST='127.0.0.1'
TASKNAME='exnet10'
DATAROOT='/mnt/data1/'+TASKNAME
UNINIT=0
INITED=1
PLANTING=1
GROWING=2
HARVESTING=3

MAX_EPOCH=5
POP_INIT=1
PRODUCT_TIMES=2
MAX_POP=5

class task:
  def __init__(self,name,epoch=0,status=UNINIT):
    self.name = name
    self.epoch = epoch
    self.status = status
  def todic(self):
    dic={"epoch":self.epoch,"status":self.status}
    return dic

class model_des:
  def __init__(self,fn='',result=0):
    self.filename=fn
    self.result=result
  
  def get_result(self):
    return self.result  

class parent_list:
  def __init__(self):
    self.obj_list=[]

  @staticmethod
  def load(filename):
    with open(filename+'/parent_list','rb') as f:
     return pickle.load(f)
  
  def save(self,path):
    filename=path+'/parent_list'
    with open(filename,'wb') as f:
      f.write(pickle.dumps(self))
    return filename

def makedirs(path):
  try:
    os.makedirs(path)
  except OSError as exc: # Python >2.5 (except OSError, exc: for Python <2.5)
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

def main(argv):
  if len(argv) > 0:
    mresult = Modresult.load(argv[0])
    print(mresult.fitness,',',mresult.acc)
    msc = ModelString.load(argv[1])
    mod = msc.load_model()
    mod.summary()
    print('training time:',msc.totaltime) 
    exit()
  for i in range(0,50):  
    epochpath=DATAROOT+'/'+str(i)
    if os.path.exists(epochpath):
      dirs = os.listdir(epochpath)
      for d in dirs:
        modelpath = os.path.join(epochpath, d)
        if os.path.isdir(modelpath):
          mresult = Modresult.load(modelpath+'/result')
          print(modelpath+':',mresult.acc,' fit:',mresult.fitness)
          modelfilepath = modelpath+'/'+d
          #print('list modelpath:',modelfilepath)
          #msc = ModelString.load(modelfilepath)
          #mod = msc.load_model()
          #mod.summary()
          #print('training time:',msc.totaltime)
if __name__ == "__main__":
  main(sys.argv[1:])
