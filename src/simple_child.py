from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf
from src.child import Child

class SimpleChild(Child):
  def __init__(self,
               images,
               labels,
               food=0,
               fitness=0.1,
               param_num=10000,
               batch_size=32,
               graph=None,
               **kwargs
              ):
    """
    """
    super(SimpleChild, self).__init__(
               images,
               labels,
               food=0,
               fitness=0.1,
               param_num=10000,
               batch_size=32,
               graph=None)

  def _model(self, images, is_training, reuse=False):
    """Compute the logits given the images."""

    is_training = True
    #with self.graph.as_default():
    #x = tf.convert_to_tensor(images)
    x = tf.reshape(images,[-1,784])
    with tf.variable_scope("simple",reuse=reuse):
      print("images:",x)
      W = tf.get_variable(name="weight",initializer=tf.random_normal([784,10],dtype=tf.float32),trainable=True)
      b = tf.get_variable(name="b",initializer=tf.zeros([10],dtype=tf.float32),trainable=True)
      y = tf.matmul(tf.cast(x,tf.float32),W)+b

    return y

  def _get_train_op(self):
    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    self.train_op = self.optimizer.minimize(
            loss=self.loss,
            global_step=self.global_step)
    

  
