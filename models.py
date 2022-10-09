import tensorflow as tf
import keras
import numpy as np
from sklearn import svm
# --------------------
# 包含模型：SVM、MLP调用函数，CNN、AlexNet、ResNet模型
# 修改时间：2022/9/26 2：13
# ------------------------

    
# CNN模型，网络结构给定
def CNN(l2_rate, dropout_rate):
    model = tf.keras.Sequential([
    #(-1,20,20,1)->(-1,20,20,32)
    tf.keras.layers.Conv2D(input_shape=(20, 20, 1),filters=32,kernel_size=3,strides=1,padding='same'),     # Padding method),
    #(-1,20,20,32)->(-1,10,10,32)
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,10,10,32)->(-1,10,10,64)
    tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'),
    #(-1,10,10,64)->(-1,5,5,64)
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,5,5,64)->(-1,5,5,64)
    tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'),
    #(-1,5,5,64)->(-1,5*5*64)
    tf.keras.layers.Flatten(),
    #(-1,5*5*64)->(-1,256)
    tf.keras.layers.Dense(256, activation=tf.nn.relu
                         ,kernel_regularizer=keras.regularizers.l2(l2_rate) # L2正则化
                         ),
    #dropout
    tf.keras.layers.Dropout(rate=dropout_rate),
    #(-1,256)->(-1,31)
    tf.keras.layers.Dense(31, activation=tf.nn.softmax)
])
    return model

# AlexNet网络
def AlexNet():
    model = tf.keras.Sequential([
    # layer_1
    tf.keras.layers.Conv2D(64,(3,3),strides=(1,1),input_shape=(20,20,1),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'),     # Padding method),
    tf.keras.layers.Conv2D(64,(3,3),strides=(1,1),padding='same',data_format='channels_last',kernel_initializer='uniform',activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    #layer_2
    tf.keras.layers.Conv2D(128,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'),
    tf.keras.layers.Conv2D(128,(2,2),strides=(1,1),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'),
    tf.keras.layers.MaxPool2D(2,2),

    #layer_3
    tf.keras.layers.Conv2D(256,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'),
    tf.keras.layers.Conv2D(256,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'),
    tf.keras.layers.Conv2D(256,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    #layer_4
    tf.keras.layers.Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'),
    tf.keras.layers.Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'),
    tf.keras.layers.Conv2D(512,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    #layer_5
    tf.keras.layers.Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'),
    tf.keras.layers.Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'),
    tf.keras.layers.Conv2D(512,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation='relu'),
    # tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation=tf.nn.relu),
    tf.keras.layers.Dense(4096, activation=tf.nn.relu),
    tf.keras.layers.Dense(1000, activation=tf.nn.relu),
    tf.keras.layers.Dense(31, activation=tf.nn.softmax)
])
    return model

# ResNet
# 3x3 convolution
def conv3x3(channels, stride=1, kernel=(3, 3)):
    return keras.layers.Conv2D(channels, kernel, strides=stride, padding='same',
                               use_bias=False,
                            kernel_initializer=tf.random_normal_initializer())

class ResnetBlock(keras.Model):
    def __init__(self, channels, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path
        self.conv1 = conv3x3(channels, strides)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = conv3x3(channels)
        self.bn2 = keras.layers.BatchNormalization()
        if residual_path:
            self.down_conv = conv3x3(channels, strides, kernel=(1, 1))
            self.down_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        residual = inputs
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # this module can be added into self.
        # however, module in for can not be added.
        if self.residual_path:
            residual = self.down_bn(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.down_conv(residual)
        x = x + residual
        return x


class ResNet(keras.Model):
  def __init__(self, block_list, num_classes, initial_filters=16, **kwargs):
      super(ResNet, self).__init__(**kwargs)
      self.num_blocks = len(block_list)
      self.block_list = block_list

      self.in_channels = initial_filters
      self.out_channels = initial_filters
      self.conv_initial = conv3x3(self.out_channels)
      self.blocks = keras.models.Sequential(name='dynamic-blocks')
      # build all the blocks
      for block_id in range(len(block_list)):
          for layer_id in range(block_list[block_id]):
              if block_id != 0 and layer_id == 0:
                  block = ResnetBlock(self.out_channels, strides=2, residual_path=True)
              else:
                  if self.in_channels != self.out_channels:
                      residual_path = True
                  else:
                      residual_path = False
                  block = ResnetBlock(self.out_channels, residual_path=residual_path)
              self.in_channels = self.out_channels
              self.blocks.add(block)
          self.out_channels *= 2
      self.final_bn = keras.layers.BatchNormalization()
      self.avg_pool = keras.layers.GlobalAveragePooling2D()
      self.fc = keras.layers.Dense(num_classes, activation='softmax')

  def call(self, inputs, training=None):
    out = self.conv_initial(inputs)
    out = self.blocks(out, training=training)
    out = self.final_bn(out, training=training)
    out = tf.nn.relu(out)
    out = self.avg_pool(out)
    out = self.fc(out)
    return out


# SVM/MLP使用的数据读取函数
def getImage(data_generator, label_list):
    x_data = []
    y_data = []
    for i in range(len(data_generator)):
        x = data_generator.next()[0].reshape((1, 400))
        y = label_list[i]
        x_data.append(x)
        y_data.append(y)
    return np.array(x_data).reshape((-1, 400)), np.array(y_data)


