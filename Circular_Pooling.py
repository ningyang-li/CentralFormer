# -*- coding: utf-8 -*-

from keras.engine.base_layer import Layer
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


class Circular_Pooling(Layer):
    '''
    Circular pooling (CP), a novel pooling pattern tailored for hyperspectral image (HSI) classification,
    Each HSI cube/patch is considered as a center pixel and some circles.
    CP just pools the circles but protects the center pixel, thereby ensuring the relevance and effectiveness of features.

    pool_size = 2, strides = 1 ==> n_circles - 1, width - 2
    pool_size = 2, strides = 2 ==> n_circles - 2, width - 4
    '''
    def __init__(self, pool_size=2, strides=1, pool_band=2, strides_band=2, pool_mode="avg", data_format="channels_first", **kwargs):
        super(Circular_Pooling, self).__init__(**kwargs)

        self.pool_size = pool_size
        self.strides = strides
        self.pool_band = pool_band
        self.strides_band = strides_band
        self.pool_mode = pool_mode
        self.data_format = data_format

        self.min_circle = 1
        self.min_width = self.min_circle * 2 + 1
        self.pooled_shape = None


    def build(self, input_shape):
        assert len(input_shape) == 5

        # sample width
        self.width = input_shape[-2]
        self.n_band = input_shape[-1]
        
        # number of circles
        self.n_circle = self.width // 2
        
        # width of pooled sample
        # new_width = int((width - 1) / 2 - self.pool_size + 1)
        if self.pool_size == 2:
            if self.pool_band == 2 and self.strides_band == 2:
                cur_width = self.width
                cur_band = self.n_band
                
                pooled_band = int(np.ceil(cur_band / 2.))
                
                if cur_width > 3:
                    pooled_width = cur_width - 2 * self.strides
                else:
                    pooled_width = cur_width

        self.pooled_shape = (input_shape[0], input_shape[1], pooled_width, pooled_width, pooled_band)
        print("input shape is:", input_shape)
        print("pooled shape is:", self.pooled_shape)

        super(Circular_Pooling, self).build(input_shape)
    

    def call(self, x):       
        if self.width <= 3:
            print("there is only one circle, STOP CP!")
            output = x
        else:
            # position of center pixels in old and new samples
            center = (self.width // 2, self.width // 2)
            
            # 9 regions
            '''
            1 | 2 | 3
            ---------
            8 | 0 | 4
            ---------     
            7 | 6 | 5
            '''
            
            # (bs, n_channle, 4, 4, n_band)
            r1 = K.pool3d(x[:, :, 0:center[0], 0:center[1], :],
                          pool_size=(self.pool_size, self.pool_size, 1),
                          strides=(self.strides, self.strides, 1),
                          padding="valid",
                          data_format=self.data_format,
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 1, n_band)
            r2 = K.pool3d(x[:, :, 0:center[0], center[1]:center[1] + 1, :],
                          pool_size=(self.pool_size, 1, 1),
                          strides=(self.strides, 1, 1),
                          padding="valid",
                          data_format=self.data_format,
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 4, n_band)
            r3 = K.pool3d(x[:, :, 0:center[0], center[1] + 1:, :],
                          pool_size=(self.pool_size, self.pool_size, 1),
                          strides=(self.strides, self.strides, 1),
                          padding="valid",
                          data_format=self.data_format,
                          pool_mode="avg")
            
            # (bs, n_channle, 1, 4, n_band)
            r8 = K.pool3d(x[:, :, center[0]:center[0] + 1, 0:center[1], :],
                          pool_size=(1, self.pool_size, 1),
                          strides=(1, self.strides, 1),
                          padding="valid",
                          data_format=self.data_format,
                          pool_mode="avg")
            
            # (bs, n_channle, 1, 1, n_band)
            r0 = x[:, :, center[0]:center[0] + 1, center[1]:center[1] + 1, :]
            
            # (bs, n_channle, 1, 4, n_band)
            r4 = K.pool3d(x[:, :, center[0]:center[0] + 1, center[1] + 1:, :],
                          pool_size=(1, self.pool_size, 1),
                          strides=(1, self.strides, 1),
                          padding="valid",
                          data_format=self.data_format,
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 4, n_band)
            r7 = K.pool3d(x[:, :, center[0] + 1:, 0:center[1], :],
                          pool_size=(self.pool_size, self.pool_size, 1),
                          strides=(self.strides, self.strides, 1),
                          padding="valid",
                          data_format=self.data_format,
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 1, n_band)
            r6 = K.pool3d(x[:, :, center[0] + 1:, center[1]:center[1] + 1, :],
                          pool_size=(self.pool_size, 1, 1),
                          strides=(self.strides, 1, 1),
                          padding="valid",
                          data_format=self.data_format,
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 4, n_band)
            r5 = K.pool3d(x[:, :, center[0] + 1:, center[1] + 1:, :],
                          pool_size=(self.pool_size, self.pool_size, 1),
                          strides=(self.strides, self.strides, 1),
                          padding="valid",
                          data_format=self.data_format,
                          pool_mode="avg")
            
            # stack
            r123 = tf.concat([r1, r2, r3], axis=3)
            r804 = tf.concat([r8, r0, r4], axis=3)
            r765 = tf.concat([r7, r6, r5], axis=3)
            
            output = tf.concat([r123, r804, r765], axis=2)
        
        # pooling in spectral dimension
        output = K.pool3d(output,
                          pool_size=(1, 1, self.pool_band),
                          strides=(1, 1, self.strides_band),
                          padding="same",
                          data_format=self.data_format,
                          pool_mode="avg")
        
        return output


    def compute_output_shape(self, input_shape):
        return self.pooled_shape

