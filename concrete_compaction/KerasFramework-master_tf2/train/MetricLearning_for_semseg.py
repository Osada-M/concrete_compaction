""" 
MIT License

Copyright (c) 2018 Takato Kimura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from tensorflow.distribute import MirroredStrategy
# import numpy as np
# import pandas as pd
# import random
# import matplotlib.pyplot as plt
# import os
# from PIL import Image
# import sys

from scipy.stats.stats import SpearmanrResult
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Layer, add, concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
# from tensorflow.math import l2_normalize

# my module
# from mymodel.SemanticSegmentation import SemanticSegmentation
# from colorPrint import Cprint as cp
# from MyUtils import Utils, Calc, TimeCounter
# from MyUtils import ImageManager as im
# from MyUtils import ModelManager as mm



class L2ConstrainLayer(tf.keras.layers.Layer):
    """
    @brief  距離学習: 特徴量の最終出力をL2正則化
    @note   参考: http://urusulambda.com/2020/08/12/l2-constrain-softmax-lossをtensorflow-kerasでmnistをとりあえず動かす/
    """
    # コンストラクタ
    def __init__(self, *args, **kwargs):
        # 親クラス(tf.keras.layers.Layer)のコンストラクタに自身のクラスを渡す?
        super(L2ConstrainLayer, self).__init__(*args, **kwargs)
        self.alpha = tf.Variable(30.)

    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1) * self.alpha
        # return l2_normalize(inputs, axis=1) * self.alpha

class ArcFace(Layer):
    """
    @class  ArcFace
    @brief  ArcFaceを行う層.
    """
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)
    
    def get_config(self):
        config = {
            'n_classes'  : self.n_classes,
            's'          : self.s,
            'm'          : self.m,
            'regularizer': self.regularizer
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # x = l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # W = l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


class SphereFace(Layer):
    """
    @class  SphereFace
    @brief  SphereFaceを行う層
    @note   フォワードを調べる.
            親クラスはLayerクラス.
    """
    def __init__(self, n_classes=2, s=30.0, m=1.35, regularizer=None, nullfication:bool=False, label_smoothing:float=0., **kwargs):
        super(SphereFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)
        self.nullfication = nullfication
        self.label_smoothing = label_smoothing
    
    def get_config(self):
        config = {
            'n_classes'  : self.n_classes,
            's'          : self.s,
            'm'          : self.m,
            'regularizer': self.regularizer
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(SphereFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # x = l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # W = l2_normalize(self.W, axis=0)
        
        ## 内積
        logits = x @ W
        
        ## SphereFaceの適用と算出
        if not self.nullfication:
            ## 特徴ベクトルと代表ベクトルとがなす角
            theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            target_logits = tf.cos(self.m * theta)
            if self.label_smoothing:
                logits = self.label_smoothing * logits * (1 - y) + (1 - self.label_smoothing) * target_logits * y
            else:
                logits = logits * (1 - y) + target_logits * y
                
        ## 値を強調
        logits *= self.s
        
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

class CosFace(Layer):
    """
    @class  CosFace
    @brief  CosFaceを行う層
    """
    def __init__(self, n_classes=10, s=30.0, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)
    
    def get_config(self):
        config = {
            'n_classes'  : self.n_classes,
            's'          : self.s,
            'm'          : self.m,
            'regularizer': self.regularizer
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # x = l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # W = l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        target_logits = logits - self.m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)
