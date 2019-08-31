import os
import sys
import pickle as pickle
import numpy as np
import tensorflow as tf

import src.utils.mnist_reader as mnist_reader

def _read_data(data_path):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
 
  batch_images, batch_labels = mnist_reader.load_mnist(data_path, kind='train')
    #X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    #data = pickle.load(finp,encoding='bytes')
  #batch_images = batch_images.astype(np.float32) / 255.0
  print(batch_images[0])
    #batch_labels = np.array(data[b'labels'], dtype=np.int32)
  images.append(batch_images)
  labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 1, 28, 28])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels

def _read_data_test(data_path):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  X_test, y_test = mnist_reader.load_mnist(data_path, kind='t10k')
    #X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    #data = pickle.load(finp,encoding='bytes')
  print(X_test)
  batch_images = X_test.astype(np.float32) / 255.0
  batch_labels = np.array(y_test, dtype=np.int32)
  images.append(batch_images)
  labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 1, 28, 28])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels

def read_data(data_path, num_valids=5000):
  print ("-" * 80)
  print ("Reading data")

  images, labels = {}, {}

  
  images["train"], labels["train"] = _read_data(data_path)

  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]

    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None

  images["test"], labels["test"] = _read_data_test(data_path)

  print ("Prepropcess: [subtract mean], [divide std]")
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print ("mean: {}".format(np.reshape(mean * 255.0, [-1])))
  print ("std: {}".format(np.reshape(std * 255.0, [-1])))

  images["train"] = (images["train"] - mean) / std
  if num_valids:
    images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels

