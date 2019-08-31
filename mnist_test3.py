import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './fashion-minst/'

def main(unused_argv):
    # Load training and eval data
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True, validation_size=0)
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_data, train_labels = shuffle(train_data, train_labels)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Reshape((28, 28,1)))
    ## out 24x24x32 ?
    
    model.add(Conv2D(32,
                 activation='relu',
                 input_shape=input_shape,
                 nb_row=5,
                 nb_col=5))
    ## out 12x12x32
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    
    ## out 8x8x64
    model.add(Conv2D(64,
              activation='relu',
              nb_row=5,
              nb_col=5))
    ## out 4x4x64
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    
    ## out 4x4x64    
    model.add(Flatten())
    
    ## out 512
    model.add(Dense(256, activation='relu'))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy'])

    for j in range(10):
      model.fit(train_data, train_labels, epochs=200, batch_size=64, verbose=1, validation_split=0.05)

      loss, accuracy = model.evaluate(eval_data, eval_labels)
      print('Turn:',j)
      print('Test loss:', loss)
      print('Accuracy:', accuracy)
   
if __name__ == "__main__":
  main(None)
    

