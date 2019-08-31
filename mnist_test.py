# -- coding: utf-8 --
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow.examples.tutorials.mnist.input_data as input_data

fashion=input_data.read_data_sets("fashion-minst/",one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.matmul(x,w)*b)


#权重初始化
def weight_Variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return  tf.Variable(initial)

def bias_Variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积层和池化层的定义

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#卷积和池化:第一层
W_conv1=weight_Variable([5,5,1,32])  #前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目,每一个输出通道都有一个对应的偏置量
b_conv1=bias_Variable([32])


x_image=tf.reshape(x,[-1,28,28,1]) #[-1,宽，高，颜色通道数] 作为卷积层的输入

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

#卷积和池化:第二层
W_conv2=weight_Variable([5,5,32,64])
b_conv2=bias_Variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
#第三层
W_conv3=weight_Variable([5,5,64,128])
b_conv3=bias_Variable([128])

h_conv3=tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
h_pool3=max_pool_2x2(h_conv3)
#第四层
W_conv4=weight_Variable([5,5,128,256])
b_conv4=bias_Variable([256])

h_conv4=tf.nn.relu(conv2d(h_pool3,W_conv4)+b_conv4)
h_pool4=max_pool_2x2(h_conv4)
#全连接层

W_fc1=weight_Variable([20*10*256,1024])  #1024个神经元的全连接层,why 10247?
b_fc1=bias_Variable([1024])

h_pool2_flat=tf.reshape(h_pool4,[-1,20*10*256])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#Dropout
keep_prob=tf.placeholder("float")#用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#输出层
W_fc2=weight_Variable([1024,10])
b_fc2=bias_Variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#训练和评估模型
#tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值,
# 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签比如tf.argmax(y,1)返回的是模型对于任一输入x预>测到的标签值，
# 而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
print("cross_entropy shape:", cross_entropy)

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_predicton=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))# tf.equal 来检测我们的预测是否真实标签匹配(索>引位置一样表示匹配)
print("correct_predicton shape:", correct_predicton)

accuracy=tf.reduce_mean(tf.cast(correct_predicton,"float"))#    将correct_predicton转换为float型
print("accuracy shape:", accuracy)


sess.run(tf.global_variables_initializer())
for i in  range(20000):

    batch=fashion.train.next_batch(50)# #按批次训练，每批50行数据
    if i%100 ==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
