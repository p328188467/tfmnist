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

from src.utils.tfutils import count_model_params

class Child(object):
  def __init__(self,
               images,
               labels,
               food=0,
               fitness=0.1,
               param_num=10000,
               batch_size=50,
               graph=None):
    print ("-" * 80)
    # cache images and labels
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    self.graph = tf.Graph()
    self.x_valid = None
    with self.graph.as_default():
        self.num_train_examples = np.shape(images["train"])[0]
        self.num_train_batches = (self.num_train_examples + self.batch_size - 1) // self.batch_size
        x_train, y_train = tf.train.shuffle_batch([images["train"], labels["train"]],  batch_size=self.batch_size,
            capacity=50000,
            enqueue_many=True,
            min_after_dequeue=0,
            num_threads=16,
            seed=None,
            allow_smaller_final_batch=True,
            )
        self.x_train = x_train
        self.y_train = y_train
        self.num_test_examples = np.shape(images["test"])[0]
        self.num_test_batches = ( (self.num_test_examples + self.batch_size - 1)  // self.batch_size)
        self.x_test, self.y_test = tf.train.batch( [images["test"], labels["test"]],
                batch_size=self.batch_size,
                capacity=10000,
                enqueue_many=True,
                num_threads=1,
                allow_smaller_final_batch=True,
            )
    
    return
  
  def build_train(self):
    print ("Build train graph")
    
    with self.graph.as_default():
      logits = self._model(self.x_train, True)
      log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(self.y_train,tf.int32))
      #y_ = tf.nn.softmax(logits)
      #print("y",y_)
      #print("s.y_",self.y_train)
      #self.loss = -tf.reduce_sum(tf.cast(self.y_train,tf.float32)*tf.log(y_),1)
      self.loss = tf.reduce_mean(log_probs)
      self.logits_view = tf.reduce_mean(logits)
      self.train_preds = tf.argmax(logits, axis=1)
      self.train_preds = tf.to_int32(self.train_preds)
      self.train_acc = tf.equal(self.train_preds, tf.cast(self.y_train,tf.int32))
      self.train_acc = tf.to_int32(self.train_acc)
      self.train_acc = tf.reduce_sum(self.train_acc)

      tf_variables = tf.trainable_variables()
      self.num_vars = count_model_params(tf_variables)
      print ("-" * 80)
      for var in tf_variables:
        print (var)

      self.global_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name="global_step")

      self._get_train_op()
    """
      def build_valid(self):
        if self.x_valid is not None:
          print ("-" * 80)
          print ("Build valid graph")
        logits = self._model(self.x_valid, False, reuse=True)
        self.valid_preds = tf.argmax(logits, axis=1)
        self.valid_preds = tf.to_int32(self.valid_preds)
        self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
        self.valid_acc = tf.to_int32(self.valid_acc)
        self.valid_acc = tf.reduce_sum(self.valid_acc)
    """
  def build_test(self):
    print ("-" * 80)
    print ("Build test graph")
    with self.graph.as_default():
      logits = self._model(self.x_test, False, reuse=True)
      tf_variables = tf.trainable_variables()
      for var in tf_variables:
        print (var)
      self.test_preds = tf.argmax(logits, axis=1)
      self.test_preds = tf.to_int32(self.test_preds)
      self.test_acc = tf.equal(self.test_preds, self.y_test)
      self.test_acc = tf.to_int32(self.test_acc)
      self.test_acc = tf.reduce_sum(self.test_acc)

  def _model(self, images, is_training, reuse=None):
    raise NotImplementedError("Abstract method")

  def _get_train_op(self):
    raise NotImplementedError("Abstract method")
