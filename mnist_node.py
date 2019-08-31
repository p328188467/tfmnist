# -*- coding:utf-8 -*-
import random
import pickle
import keras
import datetime
from keras.models import Model,  model_from_json 
from mnist_model import ExNet
from keras.layers import Input, add, Reshape, Flatten, Dense, Dropout, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.utils import plot_model
import os ,errno

def makedirs(path):
  try:
    os.makedirs(path)
  except OSError as exc: # Python >2.5 (except OSError, exc: for Python <2.5)
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else: raise

class NetNode:
  def __init__(self,name,ntype,deep,width):
    self.name=name
    self.ntype=ntype
    self.epoch=0
    self.deep=deep
    self.width=width
    self.dropout_val=0

  def deeper(self):
    self.deep=self.deep+1
    self.epoch+=1

  def wider(self):
    self.width = self.width+1
    self.epoch+=1

  def dropout_add(self):
    self.dropout_val=min(self.dropout_val+1,5)
    
  def tolayers(self,x,indim,channels):
    if self.ntype == 0:
      return ExNet.res_tolayers(x,indim,channels,self.name,self.width,self.deep,self.dropout_val)
    elif self.ntype == 1:
      return ExNet.mn_tolayers(x,indim,channels,self.name,self.width,self.deep,self.dropout_val)
    elif self.ntype == 2:
        return ExNet.conv_tolayers(x,indim,channels,self.name,self.width,self.deep,self.dropout_val)
    
  def getDescription(self):
    #利用str的format格式化字符串
    #利用生成器推导式去获取key和self中key对应的值的集合
    return ",".join("{}={}".format(key,getattr(self,key)) for key in self.__dict__.keys())

  def __str__(self):
    return "{}->({})".format(self.__class__.__name__,self.getDescription())

def listDistance(l1,l2):
  matrix = [[ 0 for j in range(len(l2) + 1)] for i in range(len(l1) + 1)]
  for i in range(1,len(l1)+1):
    matrix[i][0] = l1[i-1].epoch

  for j in range(1,len(l2)+1):
    matrix[0][j] = l2[j-1].epoch

  for i in range(1, len(l1)+1):
    for j in range(1, len(l2)+1):
      if(l1[i-1].ntype == l2[j-1].ntype ):
        d = abs(l1[i-1].epoch - l2[j-1].epoch)
      else:
        d = abs(l1[i-1].epoch + l2[j-1].epoch)
    matrix[i][j] = min(matrix[i-1][j] + l1[i-1].epoch, matrix[i][j-1]+l2[j-1].epoch, matrix[i-1][j-1]+d)

  return matrix[len(l1)][len(l2)]


  

class ModelString:
  def __init__(self,mname):
    self.nodelist=[]
    self.epoch=0
    self.rootpath=''
    self.weightsfile=''
    self.model_filename=''
    self.create_time = datetime.datetime.now()
    self.modified_time = datetime.datetime.now()
    tm = (self.modified_time - self.create_time)
    self.basename=mname
    self.modelname = mname+'_'+str(self.epoch) +'_'+ str(tm.microseconds)
    self.lastacc=0
    self.totaltime=0

  def mutated(self):
    if len(self.nodelist) == 0:
      self.nodelist.append(self.newnetNode())
      self.epoch+=1
      self.modified_time = datetime.datetime.now()
      tm = (self.modified_time - self.create_time)
      self.modelname = self.basename+'_'+ str(self.epoch) +'_'+str(tm.microseconds)
      return

    roll = random.randint(0,3)
    pos = random.randint(0,len(self.nodelist))

    if roll == 0:
      ##append node
      self.nodelist.insert(pos,self.newnetNode())
    elif roll == 1:
      ##deeper node
      self.nodelist[pos-1].deeper()
    elif roll == 2:
      ##wider node
      self.nodelist[pos-1].wider()
    elif roll == 3:
      self.nodelist[pos-1].dropout_add()

    self.epoch+=1
    self.modified_time = datetime.datetime.now()
    tm = (self.modified_time - self.create_time)
    self.modelname = self.basename+'_'+str(self.epoch) +'_'+str(tm.microseconds)
    self.lastacc = 0
    self.taotaltime = 0
    return
    
    
  def newnetNode(self):
    ntype = random.randint(0,2)
    if ntype == 0:
      nodename = 'res_'+str(self.epoch)
    elif ntype == 1:
      nodename = 'mbl_'+str(self.epoch)

    elif ntype == 2:
      nodename = 'cov_'+str(self.epoch)
  
    newnode=NetNode(nodename,ntype,1,1)
    return newnode

  def save(self):
    filename=self.rootpath+'/'+self.modelname+'/'+self.modelname
    makedirs(self.rootpath+'/'+self.modelname)
    with open(filename,'wb') as f:
      f.write(pickle.dumps(self))
    return filename

  @staticmethod
  def load(filename):
    with open(filename ,'rb') as f:
     return pickle.load(f)
  
  def getDescription(self):
    #利用str的format格式化字符串
    #利用生成器推导式去获取key和self中key对应的值的集合
    return ",".join("{}={}".format(key,getattr(self,key)) for key in self.__dict__.keys())

  def __str__(self):
    modelstr = "{}->({})".format(self.__class__.__name__,self.getDescription())
    liststr = "\n".join("{},{}".format(str(i),v.__str__()) for i,v in enumerate(self.nodelist))
    return modelstr+"\n"+liststr
  
  def buildmodel(self):
    xin=Input((28, 28, 1, ),dtype='float32',name='input_data')
    x=Reshape((28, 28,1))(xin)
    indim = 28
    outdim = 28
    channels = 32
    
    x=Conv2D(channels,(3,3),padding='same',name='initial_conv2d')(x)
    for i,v in enumerate(self.nodelist):
      print(i,indim,channels)
      x,indim,channels = v.tolayers(x,indim,channels)
      print(i,indim,channels)
    poolsize = int((indim+1)/2)
    x = MaxPooling2D(pool_size=(int(poolsize),int(poolsize)),strides=2)(x)
    #x = Flatten()(x)
    indim = int((indim-poolsize+1)/2)
    indim = indim*indim*channels
    #x = Dropout(0.5)(x)
    
    #layername = 'dense_'+str(indim)+'_'+str(channels)+str(outdim)
    #x = Dense(int(256), activation='relu',name=layername)(x)
    layername = 'gap_'+ str(indim)+'_' + str(channels)
    x = GlobalAveragePooling2D(name=layername)(x)
    
    indim = 1
    outdim =10
    layername = 'dense_'+str(indim)+'_'+str(channels)+'_'+str(outdim)
    x = Dense(10, activation='softmax',name=layername)(x)

    model = Model(xin, x, name=self.modelname)
    
    model.summary()
    makedirs(self.rootpath+'/'+self.modelname)
    modelpngname=self.rootpath+'/'+self.modelname+'/'+self.modelname+'.png'
    print('pngfile:',modelpngname)
    plot_model(model, to_file=modelpngname, show_shapes=True)
    # serialize model to JSON
    model_json = model.to_json()
    self.model_filename=self.rootpath+'/'+self.modelname+'/' +self.modelname+'.json'
    with open(self.model_filename, "w") as json_file:
      json_file.write(model_json)

  def load_model(self):
    with open(self.model_filename, "rb") as json_file:
      loaded_model_json = json_file.read()
      loaded_model = model_from_json(loaded_model_json.decode())
      return loaded_model
  
  def load_weights(self,mod,by_name=False):
    print("to load weights:",self.weightsfile)
    if os.path.exists(self.weightsfile):
      print("load weights:",self.weightsfile)
      mod.load_weights(self.weightsfile,by_name=by_name)
  
  def save_weights(self,mod):
    self.weightsfile = self.rootpath+'/'+self.modelname+'/' +self.modelname+'.h5'
    print("save weights:",self.weightsfile)
    mod.save_weights(self.weightsfile)
  
class Modresult:
  def __init__(self,acc,fitness,rootpath):
    self.acc = acc
    self.fitness=fitness
    self.rootpath=rootpath

  def save(self):
    filename=self.rootpath+'/'+'result'
    with open(filename,'wb') as f:
      f.write(pickle.dumps(self))
    return filename

  @staticmethod
  def load(filename):
    with open(filename ,'rb') as f:
     return pickle.load(f)


def main(unused_argv):
  print(keras.__version__)
  ms=ModelString('test')
  ms.mutated()
  f= ms.save()
  times=3
  while times>0:
    times=times-1
    ms1=ms.load(f)
    ms1.mutated()
    ms1.buildmodel()
    f=ms1.save()

  print(ms1)  
   
if __name__ == "__main__":
  main(None)
