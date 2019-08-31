from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pickle
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

from src.data_utils import read_data
from src.simple_child import SimpleChild

def create_popu(prarent,images,labels):
  return SimpleChild(images,labels)

def main(_):
  print("-" * 80)
  images, labels = read_data("./fashion-minst/")
  epoches=[]
  popu=create_popu(None,images,labels)
  train_child(popu)

def train_child(child):
  child.build_train()
  child.build_test()
  
  with child.graph.as_default():
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config) as sess:
      for i in range(1000):
        loss,train_acc,view,train_op =sess.run([child.loss,child.train_acc,child.logits_view,child.train_op])
        if i % 5 == 0:
          curr_time = time.time()
          log_string = ""
          log_string += " loss={:<7.3f}".format(loss)
          log_string += " acc={:<6.4f}".format(train_acc)
          log_string += " view={:<6.4f}".format(view)
          print(log_string)
    
      num_examples = child.num_test_examples
      num_batches = child.num_test_batches
      acc_op = child.test_acc
      test_pred = child.test_preds
      total_acc = 0
      total_exp = 0

      for batch_id in range(num_batches):
        pred,acc = sess.run([test_pred,acc_op])
        total_acc += acc
        total_exp += child.batch_size
        print ("{}_accuracy: {:<6.4f}".format(
          "test", float(total_acc) / total_exp))



if __name__=="__main__":
  tf.app.run()
