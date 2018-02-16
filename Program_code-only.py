# loading training and testing data from csv

import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = 
FTEST  = 

def load(test=False, cols=None):
    """
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(fname)    # pandas 读入数据

    # 将'Image'列以空格作为分隔符，转换成 np 数组
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # 只留下'Image'和给定的列
        df = df[list(cols) + ['Image']]

    print(df.count())       # 每一列包含值的个数
    df = df.dropna()        # 丢弃空行

    # 将'image' np 数组从水平一行变成垂直一列堆叠，再归一化（灰度值0-255）
    # numpy 堆叠理解见 http://blog.csdn.net/csdn15698845876/article/details/73380803
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)            # 转换成32位浮点数组（为什么）

    if not test:                        # 只有训练集有点坐标
        y = df[df.columns[:-1]].values  # 除了'Image'一栏都是训练数据
        y = (y - 48) / 48   # （把原点定在图像中心）坐标值缩放到[-1, 1]
        X, y = shuffle(X, y,            # 打乱训练数据，(X,y)依然对应
                       random_state=42) # random.seed
        y = y.astype(np.float32)        # 转换成32位浮点数组（为什么）
    else:
        y = None

    return X, y                         # 返回训练集和标记np数组

X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
      X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
      y.shape, y.min(), y.max()))


# 1-hidden-layer NN
'''
原博此处有误，learningrate应该是float而不是str
'''

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()                # 建立顺序网络（简单的网络结构）
model.add(Dense(100,                # 中间全连接层，100个神经元
                input_dim=9216))    # 单个样本X的Shape，输入图像的像素数
model.add(Activation('relu'))       # 激活函数用Relu
model.add(Dense(30))                # 输出全连接层，30单元，单个y的Shape，15个点30个坐标

sgd = SGD(lr=0.01,                  # 自定义优化器
          momentum=0.9, 
          nesterov=True)
model.compile(loss='mean_squared_error', # 损失函数
              optimizer=sgd)             # 自定义的优化器
model.summary()                          # 显示网络结构
hist = model.fit(X, y, 
                 nb_epoch=100,           # 训练轮数
                 validation_split=0.2)   # 20%的训练样本用于验证


# Visualise

from matplotlib import pyplot

pyplot.plot(hist.history['loss'],        # 训练集中训练样本损失函数
            linewidth=2, 
            label='train set')
pyplot.plot(hist.history['val_loss'],    # 训练集中验证样本损失函数
            linewidth=2,
            label='valid set')
pyplot.grid()                            # 显示刻度(x轴上的0,20,40等)
pyplot.legend()                          # 显示图例
pyplot.xlabel('epoch')                   # x轴标签
pyplot.ylabel('loss')                    # y轴标签
pyplot.ylim(1e-3, 1e-2)                  # y轴显示范围
pyplot.yscale('log')                     # y轴刻度
pyplot.show()                            # 显示图表


# 试试不同的优化方式

model1 = Sequential()                # 建立顺序网络（简单的网络结构）
model1.add(Dense(100,                # 中间全连接层，100个神经元
                 input_dim=9216))    # 单个样本X的Shape，输入图像的像素数
model1.add(Activation('relu'))       # 激活函数用Relu
model1.add(Dense(30))                # 输出全连接层，30单元，单个y的Shape，15个点30个坐标

model1.compile(loss='mean_squared_error', # 损失函数
               optimizer='sgd')           # 默认梯度下降优化器，无栋动量
model1.summary()                          # 显示网络结构
hist1 = model.fit(X, y, 
                  nb_epoch=100,           # 训练轮数
                  validation_split=0.2)   # 20%的训练样本用于验证


# Visualise

% matplotlib inline

from matplotlib import pyplot

pyplot.plot(hist.history['loss'],        # 训练集中训练样本损失函数
            linewidth=2, 
            label='train set')
pyplot.plot(hist.history['val_loss'],    # 训练集中验证样本损失函数
            linewidth=2,                 
            label='valid set')
pyplot.plot(hist1.history['loss'],       # 训练集中训练样本损失函数（无动量）
            linewidth=2, 
            label='train set without m')
pyplot.plot(hist1.history['val_loss'],   # 训练集中验证样本损失函数（无动量）
            linewidth=2,                 
            label='valid set without m')
pyplot.grid()                            # 显示刻度(x轴上的0,20,40等)
pyplot.legend()                          # 显示图例
pyplot.xlabel('epoch')                   # x轴标签
pyplot.ylabel('loss')                    # y轴标签
pyplot.ylim(1e-3, 1e-2)                  # y轴显示范围
pyplot.yscale('log')                     # y轴刻度
pyplot.show()


from matplotlib import pyplot

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

X_test, _ = load(test=True)              # load() 返回(X,None)，用_丢弃None
y_test = model.predict(X_test)           # 用训练好的模型预测X_test

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(left=0,
                    right=1,
                    bottom=0,
                    top=1,
                    hspace=0.05,
                    wspace=0.05)

for i in range(16):
    axis = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    plot_sample(X_test[i], y_test[i], axis)

pyplot.show()


# 保存模型：避免重复训练的时间

json_string = model.to_json()
open('model1_architecture.json', 'w').write(json_string)  # 模型结构信息
model.save_weights('model1_weights.h5')                   # 模型权重信息

# 载入模型

from keras.models import model_from_json
model = model_from_json(open('model1_architecture.json').read())
model.load_weights('model1_weights.h5')

# Model 2: CNN

# 输入值从9216像素转换成(1,96,96)，1表示灰度的1通道
# 此处原文通道数有错误
def load2d(test=False, cols=None):
    X, y = load(test, cols)
    X = X.reshape(-1, 96, 96, 1)            # Image 转换格式，tf为NHWC
    return X, y

# 网络结构：3个Conv2D(分别32,64,128卷积核) + 
#          2个maxPool2D(均为500神经元)
from keras.layers import Convolution2D, MaxPooling2D, Flatten
                         
# Flatten层将多维数据一维化，用来连接卷积层Conv和
# 全连接层Dense

X, y = load2d()                     # 转换数据格式
model3 = Sequential()               # 顺序网络对象

model3.add(Convolution2D(32, 3, 3,  # 32卷积核，3x3卷积
                         input_shape=(1, 96, 96))) # 输入shape（仅需这里）
model3.add(Activation('relu'))      # 激活函数
model3.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2最大值汇合

model3.add(Convolution2D(64, 2, 2)) # 64卷积核，2x2卷积(自动计算shape)
model3.add(Activation('relu'))      # 激活函数
model3.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2最大值汇合

model3.add(Convolution2D(128, 2, 2))# 128卷积核，2x2卷积
model3.add(Activation('relu'))      # 激活函数
model3.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2最大值汇合

model3.add(Flatten())               # 压平降到一维
model3.add(Dense(500))              # 全连接层，500神经元
model3.add(Activation('relu'))      # 激活函数
model3.add(Dense(500))              # 全连接层，500神经元
model3.add(Activation('relu'))      # 激活函数
model3.add(Dense(30))               # 全连接层，30神经元(15个点坐标)

sgd = SGD(lr='0.01',                # 自定义优化函数
          momentum=0.9, 
          nesterov=True)   
model3.compile(loss='mean_squared_error', # 损失函数
               optimizer=sgd)             # 优化器
model3.summary()                          # 显示网络结构
hist3 = model3.fit(X, y, 
                   epochs=1000,           # 训练轮数
                   validation_split=0.2)  # 验证比例

# 看到这里
# 数据扩增

# 水平翻转