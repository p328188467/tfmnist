import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
import numpy as np
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
from keras.preprocessing.image import ImageDataGenerator 

DATA_DIR = './fashion-minst/'



def main(unused_argv):
    # Load training and eval data
    print(keras.__version__)
    datagen = ImageDataGenerator(
    ##  featurewise_center=True,
    ##  featurewise_std_normalization=True,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2)
    ##  horizontal_flip=True)

    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True, validation_size=0)
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_data, train_labels = shuffle(train_data, train_labels)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)
    train_data = train_data.reshape(-1,28,28,1)  # Returns np.array
    eval_data = eval_data.reshape(-1,28,28,1)  # Returns np.array
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Reshape((28, 28,1)))
    ## out 24x24x32 ?
    
    model.add(SeparableConv2D(64,(5,5)))
    ## out 12x12x32
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    
    ## out 8x8x64
    model.add(SeparableConv2D(64,(5,5)))
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

    ##datagen.fit(train_data)
    
    for j in range(2):
      batches = 0
      ##model.fit(train_data, train_labels, epochs=1, batch_size=64, verbose=1, validation_split=0.05)
      ##model.summary()    
      for  x_batch, y_batch in datagen.flow(train_data, train_labels, batch_size=256):
        model.fit(x_batch, y_batch,batch_size=32)
        batches += 1
    #    break
        if batches >= np.size(train_data,0) / 256:
            # 我们需要手动打破循环，
            # 因为生成器会无限循环
            break
      loss, accuracy = model.evaluate(eval_data, eval_labels)
      print('Turn:',j)
      print('Test loss:', loss)
      print('Accuracy:', accuracy)
   
if __name__ == "__main__":
  main(None)
    

