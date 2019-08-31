# -*- coding:utf-8 -*- 
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D,DepthwiseConv2D
from keras.layers.core import Activation, Flatten,Dense, Dropout
from keras.layers import Input, add, Reshape, ReLU
from tensorflow.python import keras
from keras.models import Model
from keras.regularizers import l2#这里加入l2正则化目的是为了防止过拟合
from keras.utils.vis_utils import plot_model
import keras.backend as K
from keras.backend import relu
from keras.utils.generic_utils import CustomObjectScope


class ExNet:
  @staticmethod
  def residual_module(x, K, stride, chanDim,indim,channels, reduce=False, reg=1e-4, bnEps=2e-5, bnMom=0.9,layername=''):#结构参考Figure 12.3右图,引入了shortcut概念，是主网络的侧网络
    """
    The residual module of the ResNet architecture.
    Parameters:
      x: The input to the residual module.
      K: The number of the filters that will be learned by the final CONV in the bottlenecks.最终卷积层的输出
      stride: Controls the stride of the convolution, help reduce the spatial dimensions of the volume *without* resorting to max-pooling.
      chanDim: Define the axis which will perform batch normalization.
      reduce: Cause not all residual module will be responsible for reducing the dimensions of spatial volums -- the red boolean will control whether reducing spatial dimensions (True) or not (False).是否降维，
      reg: Controls the regularization strength to all CONV layers in the residual module.
      bnEps: Controls the ε responsible for avoiding 'division by zero' errors when normalizing inputs.防止BN层出现除以0的异常
      bnMom: Controls the momentum for the moving average.
    Return:
      x: Return the output of the residual module.
    """
    # The shortcut branch of the ResNet module should be initialize as the input(identity) data.
    outdim=indim
    shortcut = x
    # The first block of the ResNet module -- 1x1 CONVs.
#    BNlayername = 'bn1_'+ layername + str(indim) + '_' + str(channels)
#    bn1   = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom,name=BNlayername)(x)
#    act1layername = 'act1_'+ layername + str(indim) + '_' + str(channels)
#    act1  = Activation("relu",name=act1layername)(bn1)
    # Because the biases are in the BN layers that immediately follow the convolutions, so there is no need to introduce
    #a *second* bias term since we had changed the typical CONV block order, instead of using the *pre-activation* method.
#    convlayername = 'conv1_'+layername + str(indim)+ '_' + str(channels)+'_'+str(K)
#    conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg),name=convlayername)(act1)#filter=K*0.25,kernel_size=(1,1),stride=(1,1)
    

    # The second block of the ResNet module -- 3x3 CONVs.
#    BNlayername = 'bn2_'+ layername + str(indim) + '_' + str(channels)+'_'+str(K)
#    bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom,name=BNlayername)(conv1)
    
#    act2layername = 'act2_'+ layername + str(indim) + '_' + str(channels)+'_'+str(K)
#    act2 = Activation("relu",name=act2layername)(bn2)
    
    
    convlayername = 'conv2_'+layername + str(indim)+ '_' + str(channels)+'_'+ str(K)
    conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg),name=convlayername)(x)
    
    
    if reduce:
      outdim = int((indim+1)/2)
    # The third block of the ResNet module -- 1x1 CONVs.
    BNlayername = 'bn3_'+layername + str(outdim)+ '_' + str(channels)+'_'+str(K)
    
    bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
    
    act3layername = 'act3_'+ layername + str(indim) + '_' + str(channels)+'_'+str(K)
    act3 = Activation("relu",name=act3layername)(bn3)
    
    convlayername = 'conv3_'+layername + str(indim)+ '_' + str(channels)+'_'+str(K)
    conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg),name=convlayername)(act3)
    # If we would like to reduce the spatial size, apply a CONV layer to the shortcut.
    if reduce:#是否降维，如果降维的话，需要将stride设置为大于1,更改shortcut值
      shoutcutlayername = 'shortcut_'+layername + str(indim)+ '_' + str(channels)+'_'+ str(K)
      shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(x)

      # Add together the shortcut (shortcut branch) and the final CONV (main branch).
    
    addlayername = 'add1_'+layername + str(indim)+ '_' + str(channels)
    x = add([conv3, shortcut],name=addlayername)#这个与googlenet的concatenate函数不同，add函数做简单加法，concatenate函数做横向拼接.该函数仅仅将shortcut部分和非shortcut部分相加在一起

    # Return the addition as the output of the Residual module.
    return x,outdim,K
    #f(x)输出结果=conv3+shortcut
  
  @staticmethod
  def _conv_block(inputs, filters, kernel, strides, indim, channels, layername='' ): 
    """Convolution Block 
    This function defines a 2D convolution operation with BN and relu6. 
    # Arguments 
      inputs: Tensor, input tensor of conv layer. 
      filters: Integer, the dimensionality of the output space. 
      kernel: An integer or tuple/list of 2 integers, specifying the width and height of the 2D convolution window. 
      strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the width and height. Can be a single integer to specify the same value for all spatial dimensions. 
     # Returns 
      Output tensor. 
     """ 
    #with CustomObjectScope({'relu6': keras.layers.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1 
    convlayername = 'conv_'+layername + str(indim)+ '_' + str(channels)+'_'+ str(filters)+str(kernel[0])+str(strides[0])
    
    x = Conv2D(filters, kernel, padding='same', strides=strides, name=convlayername)(inputs)
    print(indim,strides)
    if strides[0] == 2:
      indim = int((indim+1)/2)
    print(indim)
    channels = filters

    bnlayername = 'bn_'+layername + str(indim)+ '_' + str(channels)
    x = BatchNormalization(axis=channel_axis,name=bnlayername)(x)
    
    aclayername = 'ac_'+layername + str(indim)+ '_' + str(channels)
    x = ReLU(6.,name = aclayername)(x)
    return x,indim,channels

  @staticmethod
  def _bottleneck(inputs, filters, kernel, t, s,  indim, channels,  r=False, layername=''): 
    """Bottleneck 
    This function defines a basic bottleneck structure. 
    # Arguments 
      inputs: Tensor, input tensor of conv layer. 
      filters: Integer, the dimensionality of the output space. 
      kernel: An integer or tuple/list of 2 integers, specifying the width and height of the 2D convolution window. 
      t: Integer, expansion factor. t is always applied to the input size. 
      s: An integer or tuple/list of 2 integers,specifying the strides of the convolution along the width and height.Can be a single integer to specify the same value for all spatial dimensions. 
      t: Boolean, Whether to use the residuals. 
    # Returns Output tensor. 
    """ 
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1 
    tchannel = K.int_shape(inputs)[channel_axis] * t 
    convblockname='covba'+layername
    x,indim,channels = ExNet._conv_block(inputs, tchannel, (1, 1), (1, 1),indim=indim,channels=channels,layername=convblockname) 
      
    dwblockname='dw_'+layername+str(indim)+str(channels)
    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same', name=dwblockname)(x) 
    print(s,indim)
    if s == 2:
      indim = int((indim+1)/2)
    print(indim)
    bnlayername = 'bn_'+layername + str(indim)+ '_' + str(channels)
    x = BatchNormalization(axis=channel_axis, name=bnlayername)(x) 
    
    aclayername = 'ac_'+layername + str(indim)+ '_' + str(channels)
    x = ReLU(6.,name=aclayername)(x)
      
    convlayername = 'conv_'+layername + str(indim)+ '_' + str(channels)+'_'+str(filters)
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same',name=convlayername)(x) 
      
    bnlayername = 'bn1_'+layername + str(indim)+ '_' + str(channels)
    x = BatchNormalization(axis=channel_axis, name=bnlayername)(x) 
    if r: 
      addlayername = 'add1_'+layername + str(indim)+ '_' + str(channels)
      x = add([x, inputs],name=addlayername) 
    return x, indim, channels

  @staticmethod
  def _inverted_residual_block(inputs, filters, kernel, t, strides, n,indim, channels, layername=''): 
    """Inverted Residual Block 
    This function defines a sequence of 1 or more identical layers. 
    # Arguments 
      inputs: Tensor, input tensor of conv layer. 
      filters: Integer, the dimensionality of the output space. 
      kernel: An integer or tuple/list of 2 integers, specifying the width and height of the 2D convolution window. 
      t: Integer, expansion factor. t is always applied to the input size. 
      s: An integer or tuple/list of 2 integers,specifying the strides of the convolution along the width and height.Can be a single integer to specify the same value for all spatial dimensions. 
      n: Integer, layer repeat times. 
    # Returns 
      Output tensor. 
    """
    bnname='botbek0_'+layername
    x,indim,channels = ExNet._bottleneck(inputs, filters, kernel, t, strides, indim, channels, layername=bnname)
    for i in range(1, n): 
      bn1name='botbek'+str(i)+'_'+layername
      x,indim,channels = ExNet._bottleneck(x, filters, kernel, t, 1, indim, channels,True, bn1name)
    
    return x,indim,channels
  
  @staticmethod
  def res_tolayers(x,indim,channels,layername,width,deep,dropout_val):
    chanDim = -1
    #4 layer per stages
    stages = (deep+3) / 3
    layersn = deep
    outdim = indim
    bnEps=2e-5
    bnMom=0.9
    channels=channels*width
    for i in range(1, int(stages)+1):
      stride = (2, 2) if i == 1 else (1, 1)
      reslayername=layername+'_'+str(i)+str(0)
      x,outdim,channels = ExNet.residual_module(x, channels, stride=stride, chanDim=chanDim, reduce=True,indim=indim,channels=channels,layername=reslayername)#进行降维
      if i<stages:
        for j in range(1,4):
          reslayername=layername+'_'+str(i)+str(j)
          x,outdim,channels = ExNet.residual_module(x, channels, stride=(1, 1), chanDim=chanDim, bnEps=bnEps, bnMom=bnMom,indim = outdim,channels=channels,layername=reslayername)
          layersn -= 1
      else:
        for j in range(1,layersn):
          reslayername=layername+'_'+str(i)+str(j)
          x,outdim,channels = ExNet.residual_module(x, channels, stride=(1, 1), chanDim=chanDim, bnEps=bnEps, bnMom=bnMom,indim = outdim,channels=channels,layername=reslayername)
          layersn -= 1
    
    if dropout_val > 0:
      x = Dropout(dropout_val/10)(x)
    return x,outdim,channels
  
  @staticmethod
  def mn_tolayers(x,indim,channels,layername,width,deep,dropout_val):

    convblockname='covb0'+layername
    inchannels = channels*width
    x,indim,channels = ExNet._conv_block(x, channels*width, (3, 3), strides=(2, 2),indim=indim,channels=channels,layername = convblockname)
    
    for i in range(1, deep+1):
      mnlayername=layername+'_irb'+str(i)
      x,indim,channels = ExNet._inverted_residual_block(x, channels, (3, 3), t=width, strides=1, n=3,indim=indim,channels=channels,layername=mnlayername)
    
    convblockname='covb1'+layername
    x,indim,channels = ExNet._conv_block(x, inchannels, (1, 1), strides=(1, 1),indim=indim,channels=channels,layername = convblockname)
    if dropout_val > 0:
      x = Dropout(dropout_val/10)(x)
    return x,indim,channels
  
  @staticmethod
  def conv_tolayers(x,indim,channels,layername,width,deep,dropout_val):
    inputch = channels 
    channels = channels*width
    for i in range(1, deep+1):
      if i == 1 :
        convblockname='covb0'+str(i)+'_'+layername
        x,indim,channels = ExNet._conv_block(x, channels, (3, 3), strides=(2, 2),indim=indim,channels=inputch,layername = convblockname)
      else:
        convblockname='covb1'+str(i)+'_'+layername
        x,indim,channels = ExNet._conv_block(x, channels, (3, 3), strides=(1, 1),indim=indim,channels=inputch,layername = convblockname)
    if dropout_val > 0:
      x = Dropout(dropout_val/10)(x)

    return x, indim, channels
