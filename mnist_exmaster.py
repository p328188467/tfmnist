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


#REDISHOST='10.200.71.206'
REDISHOST='127.0.0.1'
TASKNAME='exnet13'
DATAROOT='/mnt/data1/'+TASKNAME
UNINIT=0
INITED=1
PLANTING=1
GROWING=2
HARVESTING=3

MAX_EPOCH=60
POP_INIT=3
PRODUCT_TIMES=1
MAX_POP=10

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

def main(unused_argv):
  pool = redis.ConnectionPool(host=REDISHOST, port=6379)
  r = redis.Redis(connection_pool=pool)
  #get task status
  ret = r.hmget(TASKNAME,'epoch','status')
  print(ret)
  t=task('temp')
  if ret[0] == None :
    t=task(TASKNAME)
    t.status=INITED
    r.hmset(TASKNAME,t.todic())
    makedirs(DATAROOT)
  else:
    t=task(TASKNAME,int(ret[0]),int(ret[1]))

  epochtimes=MAX_EPOCH
  while t.epoch < epochtimes:
    if t.status==INITED:
      epochpath=DATAROOT+'/'+str(t.epoch)
      makedirs(epochpath)
      t.status = PLANTING
      r.hmset(TASKNAME,t.todic())
    
    if t.status==PLANTING:
      ##PLANTING
      ##get parent
      pl = parent_list()
      if t.epoch == 0 :
        for i in range(0,POP_INIT):
          ##create model
          ms = ModelString('ex1')
          ms.rootpath = DATAROOT+'/'+str(t.epoch)
          makedirs(ms.rootpath)
          ms.mutated()
          ms.mutated()
          ms.mutated()
          ms.mutated()
          ms.buildmodel()
          f=ms.save()
      else:
        lastepochpath = DATAROOT+'/'+str(t.epoch-1)
        pl = parent_list.load(lastepochpath)
        for m in pl.obj_list:
          
          ms = ModelString.load(m.filename)
          ms.rootpath = DATAROOT+'/'+str(t.epoch)
          makedirs(ms.rootpath)
          ###copy parent
          ms.buildmodel()
          f=ms.save()
          
          ###new chlid
          for i in range(0,PRODUCT_TIMES):
            msc = ModelString.load(m.filename)
            msc.rootpath = DATAROOT+'/'+str(t.epoch)
            msc.mutated()
            msc.buildmodel()
            f=msc.save()
      t.status = GROWING
      r.hmset(TASKNAME,t.todic())
    
    if t.status==GROWING:
      epochpath=DATAROOT+'/'+str(t.epoch)
      dirs = os.listdir(epochpath)
      for d in dirs:
        modelpath = os.path.join(epochpath, d)
        if os.path.isdir(modelpath):
          while True:
            time.sleep(5)
            print("checking:",modelpath)
            if os.path.exists(modelpath+'/'+'result'):
              break
      t.status = HARVESTING
      r.hmset(TASKNAME,t.todic())
    
    if t.status==HARVESTING:
      ####find pop
      epochpath=DATAROOT+'/'+str(t.epoch)
      dirs = os.listdir(epochpath)
      poplist=[]
      for d in dirs:
        modelpath = os.path.join(epochpath, d)
        if os.path.isdir(modelpath):
          mresult = Modresult.load(modelpath+'/result')
          modfilepath = modelpath+'/'+d
          poplist.append((modfilepath,mresult))
          
      ####checkout parent
      poplist.sort(key=lambda e:e[1].fitness,reverse = True)
      print(poplist)
      popnum=len(poplist)
      plnew = parent_list()
      if popnum>=MAX_POP:
        ###ONLY POP_INIT POP
        topp = poplist[0]
        plnew.obj_list.append(model_des(topp[0],topp[1].fitness))
        for p in poplist[1:]:
          ms = ModelString.load(p.filename)
          ispushed = True
          for t in plnew.obj_list:
            tms = ModelString.load(t.filename)
            dis = listDistance(ms.nodelist,tms.nodelist)
            if dis < 4:
              ispushed = False
          if ispushed and len(plnew.obj_list) < POP_INIT:
            plnew.obj_list.append(model_des(p[0],p[1].fitness))
      elif popnum>=MAX_POP/PRODUCT_TIMES+1:
        parentnum = MAX_POP/PRODUCT_TIMES+1
        for p in poplist[0:parentnum]:
          plnew.obj_list.append(model_des(p[0],p[1].fitness))
      else :
        for p in poplist:
          plnew.obj_list.append(model_des(p[0],p[1].fitness))
      
      plnew.save(epochpath)
      t.status = PLANTING
      t.epoch += 1
      r.hmset(TASKNAME,t.todic())

if __name__ == "__main__":
  main(None)
