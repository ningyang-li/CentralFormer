# -*- coding: utf-8 -*-

from keras.layers import Input, Multiply, Conv3D, Concatenate, Lambda, Flatten, Dense
from keras.layers import Activation, Dropout, DepthwiseConv2D, Reshape, GlobalAveragePooling3D
from keras.layers import BatchNormalization, Add, AveragePooling3D, Permute
from keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

from Network import Network
from Adaptive_Sum import Adaptive_Sum


class CentralFormer(Network):
    '''
    Centralized spectral-spatial transformer with adaptive relevancy estimator 
    and annular pooling for hyperspectral image classification
    '''
    def __init__(self, input_shape, n_category, n_encoder=3, n_head=4, n_filters = np.array([6, 4], dtype="int"),
                 cp_pool_size=2, cp_strides=1, cp_pool_band=2, cp_strides_band=2, scheme="centralformer"):
        super().__init__("CentralFormer", input_shape, n_category)
        
        self.n_encoder = n_encoder
        self.n_head = n_head
        self.n_filters = n_filters
        self.cp_pool_size = cp_pool_size
        self.cp_strides = cp_strides
        self.cp_pool_band = cp_pool_band
        self.cp_strides_band = cp_strides_band
        self.scheme = scheme
        
        self.CP_shapes = []
        self._compute_CP_output_variance()
        
        self.build_model(self.scheme)
    
    
    # Here we defined CP method inside CentralFormer class.
    # An independent CP class file is provided in the same folder.
    def _compute_CP_output_variance(self):
        # sample width
        self.width = self.input_shape[-2]
        self.n_band = self.input_shape[-1]
        
        # number of circles
        self.n_circle = self.width // 2
        
        # width of pooled sample
        # new_width = int((width - 1) / 2 - self.pool_size + 1)
        if self.cp_pool_size == 2:
            if self.cp_pool_band == 2 and self.cp_strides_band == 2:
                cur_width = self.width
                cur_band = self.n_band
                for e in range(self.n_encoder):
                    cur_band = int(np.ceil(cur_band / 2.))
                    if cur_width > 3:
                        self.CP_shapes.append((cur_width - 2 * self.cp_strides,
                                               cur_width - 2 * self.cp_strides,
                                               cur_band))
                        cur_width -= 2
                    else:
                        self.CP_shapes.append((cur_width, cur_width, cur_band))
    
    
    def _circular_pooling(self, x, pool_size=2, strides=1, pool_band=2, strides_band=2):
        '''
        circular pooling
        pool_size = 2, strides = 1 ==> n_circles - 1, width - 2
        pool_size = 2, strides = 2 ==> n_circles - 2, width - 4
        '''
        assert self.data_format == "channels_first"
        
        shape = K.int_shape(x)
        # ensure the spatial dimension of input is a square 
        print("original shape:", shape)
        
        # sample width
        width = shape[-2]
        
        if width <= 3:
            print("there is only one circle, STOP CP!")
            output = x
        else:
            # position of center pixels in old and new samples
            center = (width // 2, width // 2)
            
            
            # 9 regions here
            '''
            1 | 2 | 3
            ---------
            8 | 0 | 4
            ---------     
            7 | 6 | 5
            '''
            
            # (bs, n_channle, 4, 4, n_band)
            r1 = K.pool3d(x[:, :, 0:center[0], 0:center[1], :],
                          pool_size=(pool_size, pool_size, 1),
                          strides=(strides, strides, 1),
                          padding="valid",
                          data_format="channels_first",
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 1, n_band)
            r2 = K.pool3d(x[:, :, 0:center[0], center[1]:center[1] + 1, :],
                          pool_size=(pool_size, 1, 1),
                          strides=(strides, 1, 1),
                          padding="valid",
                          data_format="channels_first",
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 4, n_band)
            r3 = K.pool3d(x[:, :, 0:center[0], center[1] + 1:, :],
                          pool_size=(pool_size, pool_size, 1),
                          strides=(strides, strides, 1),
                          padding="valid",
                          data_format="channels_first",
                          pool_mode="avg")
            
            # (bs, n_channle, 1, 4, n_band)
            r8 = K.pool3d(x[:, :, center[0]:center[0] + 1, 0:center[1], :],
                          pool_size=(1, pool_size, 1),
                          strides=(1, strides, 1),
                          padding="valid",
                          data_format="channels_first",
                          pool_mode="avg")
            
            # (bs, n_channle, 1, 1, n_band)
            r0 = x[:, :, center[0]:center[0] + 1, center[1]:center[1] + 1, :]
            
            # (bs, n_channle, 1, 4, n_band)
            r4 = K.pool3d(x[:, :, center[0]:center[0] + 1, center[1] + 1:, :],
                          pool_size=(1, pool_size, 1),
                          strides=(1, strides, 1),
                          padding="valid",
                          data_format="channels_first",
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 4, n_band)
            r7 = K.pool3d(x[:, :, center[0] + 1:, 0:center[1], :],
                          pool_size=(pool_size, pool_size, 1),
                          strides=(strides, strides, 1),
                          padding="valid",
                          data_format="channels_first",
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 1, n_band)
            r6 = K.pool3d(x[:, :, center[0] + 1:, center[1]:center[1] + 1, :],
                          pool_size=(pool_size, 1, 1),
                          strides=(strides, 1, 1),
                          padding="valid",
                          data_format="channels_first",
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 4, n_band)
            r5 = K.pool3d(x[:, :, center[0] + 1:, center[1] + 1:, :],
                          pool_size=(pool_size, pool_size, 1),
                          strides=(strides, strides, 1),
                          padding="valid",
                          data_format="channels_first",
                          pool_mode="avg")
            
            # stack
            r123 = tf.concat([r1, r2, r3], axis=3)
            r804 = tf.concat([r8, r0, r4], axis=3)
            r765 = tf.concat([r7, r6, r5], axis=3)
            
            output = tf.concat([r123, r804, r765], axis=2)
        
        # pooling in spectral dimension
        output = K.pool3d(output,
                          pool_size=(1, 1, pool_band),
                          strides=(1, 1, strides_band),
                          padding="same",
                          data_format="channels_first",
                          pool_mode="avg")
        
        print("pool_size:", pool_size, ", strides:", strides, ", pool_band:",
              pool_band, ", strids_band:", strides_band)
        print("after CP, shape is", K.int_shape(output), "\n")
        
        return output
        
    
    def _extract_and_expand(self, x, b_expand=True, pos=None, flag="center"):
        '''
        EE module
        
        x: 5-d tensor (n_sp, n_channel, n_row, n_col, n_band)
        extract the spectral on the position of "pos"
        
        if 'b_expand' is True,  the output will has the same shape as 'x'
        '''
        n_channel, n_row, n_col, n_band = K.int_shape(x)[1:]
        
        if flag == "center":
            if pos == None:
                # default is the center
                pos = (n_row // 2, n_col // 2)
            
            center = x[:, :, pos[0]:pos[0] + 1, pos[1]:pos[1] + 1, :]
            if b_expand:
                center = K.tile(center, (1,1,n_row,n_col,1))
            
            return center
        
        elif flag == "diagonal":
            diagonal = []
            for r in range(n_row):
                diagonal.append(x[:, :, r, r:r + 1, :])
            diagonal = tf.stack(diagonal, axis=2)
            
            diagonal = K.tile(diagonal, (1,1,1,n_col,1))
            
            return diagonal
        
        elif flag == "copy":
            return x
        
        else:
            return 0
        
    
    def spectral_relevancy(self, x):
        '''
        value range: [0, 1]
        '''
        n_row, n_col, n_band = K.int_shape(x)[-3:]
        
        # channel compression
        if K.int_shape(x)[1] > 1:
            x = Conv3D(filters=1, kernel_size=1, strides=1, data_format=self.DT, use_bias=False)(x)
        
        # extract
        c = Lambda(self._extract_and_expand, output_shape=(1,n_row,n_col,n_band),
                   arguments={"b_expand":True})(x)
        
        x = Lambda(self._extract_and_expand, output_shape=(1,n_row,n_col,n_band),
                   arguments={"flag":"copy"})(x)
        
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # cosine similarity
        # element-wise operations

        ewo_cx = Multiply()([c, x])
        ewo_cc = Multiply()([c, c])
        ewo_xx = Multiply()([x, x])
        
        def _sum_bands(x):
            return K.sum(x, axis=-1, keepdims=True)
        
        _sum_cx = Lambda(_sum_bands, output_shape=(1,n_row,n_col,1))(ewo_cx)
        _sum_cc = Lambda(_sum_bands, output_shape=(1,n_row,n_col,1))(ewo_cc)
        _sum_xx = Lambda(_sum_bands, output_shape=(1,n_row,n_col,1))(ewo_xx)
        
        # similarity
        def _sim_cos(x):
            center_x = x[:, 0:1]
            center = x[:, 1:2]
            x = x[:, 2:]
            
            center_sqrt = K.sqrt(center)
            x_sqrt = K.sqrt(x)
            sim = center_x / (center_sqrt * x_sqrt)
            sim = 1. - sim
            
            return sim

        cx_c_x = Concatenate(axis=1)([_sum_cx, _sum_cc, _sum_xx])
        spe_relevancy = Lambda(_sim_cos, output_shape=(1,n_row,n_col,1))(cx_c_x)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        # linear transformation (shrink)
        # [-1, 1] --> [0, 1]
        def _shrink(x, a1=-1., b1=1., a2=0., b2=1.):
            return ((x - a1) / (b1 - a1)) * (b2 - a2) + a2
        
        spe_relevancy = Lambda(_shrink, output_shape=(1,n_row,n_col,1))(spe_relevancy)
        
        return spe_relevancy
    
    
    def spatial_relevancy(self, x, I_gaussian):
        '''
        value range: [0, 1]
        '''
        n_row = K.int_shape(x)[-3]
        
        # # offset and reorder    
        # gaussian_2d distance
        def _gaussian_2d(x, I_gaussian):
            '''
            2-D gaussian function
            '''
            n_row = K.int_shape(x)[-3]
            n_col = n_row
            center = (n_row // 2, n_col // 2)
            
            # expansion
            r = I_gaussian[:, :, 0:n_row, 0:n_col, :]
            r = K.tile(r, (1, 1, 1, n_row, 1))
            c = K.permute_dimensions(r, (0, 1, 3, 2, 4))
            
            # computing
            pos_x_2 = K.pow(r - center[0], 2)
            pos_y_2 = K.pow(c - center[1], 2)
            
            w = K.exp((pos_x_2 + pos_y_2) / 2.)
            
            return w
        
        spa_relevancy = Lambda(_gaussian_2d, output_shape=(1, n_row, n_row, 1),
                               arguments={"I_gaussian": I_gaussian})(x)
            
        return spa_relevancy
    
    
    def _neutral_fuse(self, x):
        return (x[0] + x[1]) / 2.
      

    def ARE(self, x, I_gaussian):
        '''
        Adaptive Relevancy Estimator
        input:  x, I_gaussian
        output: relevancy mask
        '''
        n_row, n_col = K.int_shape(x)[-3:-1]
        
        # relevancy
        spe_rlv = self.spectral_relevancy(x)
        spa_rlv = self.spatial_relevancy(x, I_gaussian)
        
        # adaptive fuse
        rlv = Concatenate(axis=-1)([spe_rlv, spa_rlv])
        relevancy = Adaptive_Sum(output_dim=(1, n_row, n_col, 1))(rlv)
        
        return relevancy
    
    
    def centralization(self, x):
        # EE module
        diagonal_cube = Lambda(self._extract_and_expand, output_shape=K.int_shape(x)[1:],
                             arguments={"b_expand": True, "flag": "diagonal"})(x)
        
        # subtract
        def _subtract(x):
            return x[0] - x[1]
        
        s = Lambda(_subtract, output_shape=K.int_shape(x)[1:])([diagonal_cube, x])
        
        # gaussian activation function
        def _gaussian_1d(x):
            return K.exp(-(K.pow(x, 2)))
        
        _centralization = Lambda(_gaussian_1d, output_shape=K.int_shape(x)[1:])(s)
                
        return _centralization
    
    
    def CDPA(self, q, r, v):
        n_head, n_row, n_col, n_band = K.int_shape(q)[1:]
        
        # dot-product
        def _matmul(x):
            m = x[0] * x[1]
            m = K.sum(m, axis=-1, keepdims=True)
            return m
        
        P = Lambda(_matmul, output_shape=(n_head, n_row, n_col, 1))([q, r])

        # centralization
        C = self.centralization(P)
        
        # scale
        def _scale(x):
            return x / n_band
        
        w = Lambda(_scale, output_shape=(n_head, n_row, n_col, 1))(C)
        
        # softmax
        w = Activation("softmax")(w)
        
        # dot-product
        def _matmul2(x):
            m = x[0] * x[1]
            return m
        
        AM = Lambda(_matmul2, output_shape=(n_head, n_row, n_col, n_band))([w, v])
        
        return AM
    
    
    def CDPA2(self, q, r, v):
        n_head, n_row, n_col, n_band = K.int_shape(q)[1:]
        
        # dot-product
        def _matmul(x):            
            n_head, n_row, n_col, n_band = K.int_shape(x[0])[1:]
            
            x1 = Reshape((n_head, n_row * n_col, n_band))(x[0])
            x2 = Reshape((n_head, n_row * n_col, n_band))(x[1])
            x2 = Permute((1, 3, 2))(x2)
            
            m = tf.matmul(x1, x2)
            m = Reshape((n_head, n_row * n_col, n_row * n_col, 1))(m)
            
            return m
        
        P = Lambda(_matmul, output_shape=(n_head, n_row * n_col, n_row * n_col, 1))([q, r])

        # centralization
        C = self.centralization(P)
        
        # scale
        def _scale(x):
            return x / n_band
        
        w = Lambda(_scale, output_shape=(n_head, n_row * n_col, n_row * n_col, 1))(C)
        
        # softmax
        w = Activation("softmax")(w)
        
        # dot-product
        def _matmul2(x):
            m = tf.matmul(x[0], x[1])
            
            return m
        
        w = Reshape((n_head, n_row * n_col, n_row * n_col))(w)
        v = Reshape((n_head, n_row * n_col, n_band))(v)
        AM = Lambda(_matmul2, output_shape=(n_head, n_row, n_col, n_band))([w, v])
        AM = Reshape((n_head, n_row, n_col, n_band))(AM)
        
        return AM
    
    
    def _swish(self, x):
        return tf.nn.swish(x)
        # return K.relu(x)
    
    
    def CMHA(self, x, rlv, n_head, kernel_size=(3,3,7)):
        '''
        centralized multi-head self-attention module
        input: x, rlv
        output: F
        '''
        n_channel, n_row, n_col, n_band = K.int_shape(x)[1:]
        
        # relevancy masking
        def _repeat(x):
            return K.tile(x, (1, 1, 1, 1, 1))
        
        def _multiply(x):
            return x[0] * x[1]
        
        rlv = Lambda(_repeat, output_shape=(1, n_row, n_col, 1))(rlv)

        _R = Lambda(_multiply, output_shape=(n_channel, n_row, n_col, n_band))([x, rlv])
        
        # linear layers
        Q = Conv3D(filters=n_head, kernel_size=1, strides=1, padding="same", data_format=self.DT)(x)
        R = Conv3D(filters=n_head, kernel_size=1, strides=1, padding="same", data_format=self.DT)(_R)
        V = Conv3D(filters=n_head, kernel_size=1, strides=1, padding="same", data_format=self.DT)(x)
        
        # CDPA module
        AM = self.CDPA2(Q, R, V)
        
        # linear layer
        A = Conv3D(filters=n_head, kernel_size=1, strides=1, data_format=self.DT)(AM)
        A = BatchNormalization(axis=1)(A)
        A = Lambda(self._swish, output_shape=K.int_shape(A)[1:])(A)
        
        return A
    
    
    def depthwise_conv3d(self, x, filters, kernel_size):
        '''
        ourself depethwise conv3d
        for example, shape of input: (bs, 32, 5, 5, 200) channels_first
        for spectral dimension, kernel_size should be (1, 1, k)
        for spatial dimension, kernel_size should be (k, k, 1)
        '''
        n_channel, n_row, n_col, n_band = K.int_shape(x)[1:]
        
        return Conv3D(filters, kernel_size, strides=1, padding="same", use_bias=False, data_format=self.DT)(x)
        
    
    def inverted_residual_block(self, x, expanded_channels, output_channels, kernel_size):
        '''
        from MobileVit
        '''
        # PWC3D
        me = self.depthwise_conv3d(x, expanded_channels,  (1, 1) + kernel_size[-1:])
        
        # BWC3D
        ma = self.depthwise_conv3d(x, expanded_channels, kernel_size[:2] + (1,))
        
        m = Concatenate(axis=1)([me, ma])
        m = Conv3D(output_channels, 1, padding="same", use_bias=False, data_format=self.DT)(m)
        m = BatchNormalization(axis=1)(m)
        m = Lambda(self._swish, output_shape=K.int_shape(m)[1:])(m)
            
        return m
    

    def SSLR(self, x, n_filters):
        '''
        spectral-spatial local represetation module
        63 64 84
        '''        
        L = self.inverted_residual_block(x, n_filters[0], n_filters[1], (3, 3, 7))
        
        return L


    def CERF(self, m1, m2_, i):
        '''
        cross-encoder relevancy fusion
        '''
        n_row_2 = K.int_shape(m2_)[2]
        
        print("CERF block")
        m1 = Lambda(self._circular_pooling, output_shape=(1,) + self.CP_shapes[i],
                    arguments={"pool_size": self.cp_pool_size, "strides": self.cp_strides,
                               "pool_band": 1, "strides_band": 1})(m1)

        def _concat(x):
            return tf.concat(x, axis=-1)
        m = Lambda(_concat, output_shape=(1, n_row_2, n_row_2, 1))([m2_, m1])
        m = Adaptive_Sum(output_dim=(1, n_row_2, n_row_2, 1))(m)
        
        return m
    

    def encoder(self, x, rlv, n_head, n_filters, i, b_final=False):
        if K.int_shape(x)[1] != n_head:
            x = Conv3D(n_head, 1, data_format=self.DT, use_bias=False)(x)
        
        # SSLR module
        L = self.SSLR(x, n_filters)
        
        # CMHA module
        A = self.CMHA(L, rlv, n_head)
        
        # fusion
        F = Concatenate(axis=1)([A, L])
        F = Conv3D(n_head, 1, data_format=self.DT, use_bias=False)(F)
        F = Lambda(self._swish, output_shape=K.int_shape(F)[1:])(F)
        F = Add()([F, x])
                
        shape = K.int_shape(F)

        new_shape = (shape[1],) + self.CP_shapes[i]
        F = Lambda(self._circular_pooling, output_shape=new_shape,
                   arguments={"pool_size": 2, "strides": 1})(F)
        
        return F


    def build_model(self, scheme="centralformer"):
        # input
        I = Input(shape=self.input_shape)

        # an input puppet used to provide the width of input and generate the Gaussian 2-d distribution map
        # we did it due to the limitation of keras and tensorflow frameworks
        shape_gaussian = (1, self.input_shape[2], 1, 1)
        I_gaussian = Input(shape=shape_gaussian)
        
        # conv embedding
        embedding = Conv3D(filters=1, kernel_size=(1, 1, 3), padding="same",
                            use_bias=False, data_format=self.DT)(I)
        
        rate = [1, 2, 3]
        i = 0
        print(1)
        # ARE1
        M1 = self.ARE(embedding, I_gaussian)
        # encoder1
        F1 = self.encoder(embedding, M1, self.n_head * rate[i], self.n_filters * rate[i], i)
        
        i += 1
        print(2)
        # ARE2
        M2_ = self.ARE(F1, I_gaussian)
        # CP-RF
        M2 = self.CERF(M1, M2_, i)
        # encoder2
        F2 = self.encoder(F1, M2, self.n_head * rate[i], self.n_filters * rate[i], i)
        
        i += 1
        print(3)
        # ARE3
        M3_ = self.ARE(F2, I_gaussian)
        # CP-RF
        M3 = self.CERF(M2, M3_, i)
        # encoder3
        F3 = self.encoder(F2, M3, self.n_head * rate[i], self.n_filters * rate[i], i)
            
        # Cls
        F = Flatten()(F3)
        F = Dropout(0.5)(F)
        y = Dense(self.n_category, activation="softmax")(F)
        
        self.model = Model(inputs=[I, I_gaussian], outputs=y, name="centralformer")

