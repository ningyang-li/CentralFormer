'''
Title:    CentralFormer: Centralized Spectral-Spatial Transformer for Hyperspectral Image Classification with Adaptive Relevance Estimation and Circular Pooling
Authors:  Ningyang Li, Zhaohui Wang, Faouzi Alaya Cheikh

Environment configuration:
Keras 2.6.0
Tensorflow 2.6.0
Python 3.9.16

batch size: 32
epoch: 200
optimizer: RMSprop (learning rate: 0.001, beta1: 0.9, beta2: 0.999)
decay rate: decrease 50% every 40 epoches
'''


# packages

import argparse
from keras.layers import Input, Multiply, Conv3D, Concatenate, Lambda, Flatten, Dense
from keras.layers import Activation, Dropout, DepthwiseConv2D, Reshape, GlobalAveragePooling3D
from keras.layers import BatchNormalization, Add, AveragePooling3D, Permute
from keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

from keras import initializers, constraints, regularizers 
from keras.engine.base_layer import Layer
from keras.utils.generic_utils import to_list
import pickle
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

# parameters
parser = argparse.ArgumentParser(description="CentralFormer")

parser.add_argument("--env", type=int, default=0, help="type of operation system (0: windows, 1: ubuntu)")
parser.add_argument("--ds_name", type=str, default="Indian_Pines", help="name of data set")

parser.add_argument("--train_ratio", type=dict, default={"Indian_Pines": 0.05, "PaviaU": 0.02, "Loukia": 0.05, "Dioni": 0.05}, help="training sample proportion")
parser.add_argument("--val_ratio", type=dict, default={"Indian_Pines": 0.05, "PaviaU": 0.05, "Loukia": 0.05, "Dioni": 0.05}, help="training sample proportion")
parser.add_argument("--width", type=dict, default={"Indian_Pines": 11, "PaviaU": 5, "Loukia": 7, "Dioni": 5}, help="width of sample")
parser.add_argument("--n_category", type=dict, default={"Indian_Pines": 16, "PaviaU": 9, "Loukia": 14, "Dioni": 12}, help="number of category")
parser.add_argument("--band", type=dict, default={"Indian_Pines": 200, "PaviaU": 103, "Loukia": 176, "Dioni": 176}, help="number of band")

parser.add_argument("--bs", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=200, help="number of iteration")
parser.add_argument("--exp", type=int, default=10, help="number of experiments")

parser.add_argument("--cp_pool_size", type=int, default=2, help="pool_size of CP layer")
parser.add_argument("--cp_strides", type=int, default=1, help="strides of CP layer")
parser.add_argument("--cp_pool_band", type=int, default=2, help="pool_size of CP layer in spectral dimension")
parser.add_argument("--cp_strides_band", type=int, default=2, help="strides of CP layer in spectral dimension")
parser.add_argument("--expansion_ratio", type=float, default=1.5, help="expansion ratio of SSLR module")
parser.add_argument("--n_head", type=list, default=[4, 8, 12], help="number of attention heads")
parser.add_argument("--n_filter", type=list, default=[[6, 4], [12, 8], [18, 12]], help="number of filters of SSLR module")
parser.add_argument("--n_encoder", type=int, default=3, help="number of central encoders")

parser.add_argument("--data_format", type=str, default="channels_first", help="data format of tensor")

args = parser.parse_args()
args.ds_name = "Indian_Pines"



class Adaptive_Sum(Layer):
    '''
    sum a and b adaptively
    output = alpha * a + (1 - alpha) * b
    '''
    def __init__(self, output_dim, alpha_initializer=initializers.Constant(0.5),
                 alpha_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0),
                 alpha_regularizer=None, shared_axes=(1,2,3,4), **kwargs):
        super(Adaptive_Sum, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        if shared_axes is None:
            self.shared_axes = None
        else:
            self.shared_axes = to_list(shared_axes, allow_tuple=True)
        
        
    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True
        
        self.alpha = self.add_weight(name="alpha", shape=param_shape,
                                    initializer=self.alpha_initializer,
                                    regularizer=self.alpha_regularizer,
                                    constraint=self.alpha_constraint,
                                    trainable=True)

        super(Adaptive_Sum, self).build(input_shape)
    
    
    def call(self, x):
        a = x[:, :, :, :, 0:1]
        b = x[:, :, :, :, 1:]
        
        if K.backend() == "theano":
            return K.pattern_broadcast(self.alpha, self.param_broadcast) * a + (K.pattern_broadcast(1 - self.alpha, self.param_broadcast)) * b
        else:
            return self.alpha * a + (1 - self.alpha) * b
        
    
    def compute_output_shape(self, input_shape):
        return self.output_dim
        


# network base class
class Network():
    '''
    network base class
    '''
    def __init__(self, name, input_shape=(1, 1, 1), n_category=1, data_format="channels_first"):
        self.name = name
        print("this is " + self.name + " model!")
        if len(input_shape) == 3:
            self.n_channel, self.n_row, self.n_col = input_shape
        elif len(input_shape) == 4:
            self.n_channel, self.n_row, self.n_col, self.n_band = input_shape
        elif len(input_shape) == 2:
            self.n_channel, self.n_band = input_shape
        else:
            pass
        
        self.input_shape, self.n_category = input_shape, n_category
        self.DT = self.data_format = data_format
        

    def build_model(self):
        print(self.name + " build success")
        

    def summary(self):
        print(self.model.summary())
        

    # use these functions to save and load weights when h5py 3.0.0 package is unavailable.
    def save_weights(self, filepath):
        w = self.model.get_weights()
        with open(filepath, "wb") as file:
            pickle.dump(w, file)
        print("save success!")
        

    def load_weights(self, filepath):
        with open(filepath, "rb") as file:
            w = pickle.load(file)
        self.model.set_weights(w)
        print("load success!")



class CentralFormer(Network):
    '''
    Centralized spectral-spatial transformer for hyperspectral image classification with adaptive relevancy estimation and circular pooling
    '''
    def __init__(self, input_shape, n_category, n_encoder=args.n_encoder, n_head=args.n_head,
                n_filter=args.n_filter, cp_pool_size=args.cp_pool_size,cp_strides=args.cp_strides,
                cp_pool_band=args.cp_pool_band, cp_strides_band=args.cp_strides_band, data_format=args.data_format):
        super().__init__("CentralFormer", input_shape, n_category, data_format)
        
        # architecture parameters
        self.n_encoder = n_encoder
        self.n_head = n_head
        self.n_filter = n_filter
        self.cp_pool_size = cp_pool_size
        self.cp_strides = cp_strides
        self.cp_pool_band = cp_pool_band
        self.cp_strides_band = cp_strides_band  
        
        # compute shape of output after each CP layer
        self.CP_shapes = []
        self._compute_CP_output_variance()
        
        self.build_model()
    

    def _extract_and_expand(self, x, b_expand=True, pos=None, flag="center"):
        '''
        EE operator
        
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
            return x
        

    def _swish(self, x):
        return tf.nn.swish(x)


    def depthwise_conv3d(self, x, filters, kernel_size):
        '''
        ourself depethwise conv3d
        for example, shape of input: (bs, 32, 5, 5, 200) channels_first
        for spectral dimension, kernel_size should be (1, 1, k), pixel-wise
        for spatial dimension, kernel_size should be (k, k, 1), band-wise
        '''
        n_channel, n_row, n_col, n_band = K.int_shape(x)[1:]

        return Conv3D(filters, kernel_size, strides=1, padding="same", use_bias=False, data_format=self.DT)(x)


    def inverted_residual_block(self, x, expanded_channels, output_channels, kernel_size):
        '''
        inspired by MobileVit
        '''
        # pixel-wise conv3d
        Le = self.depthwise_conv3d(x, expanded_channels,  (1, 1) + kernel_size[-1:])

        # band-wise conv3d
        La = self.depthwise_conv3d(x, expanded_channels, kernel_size[:2] + (1,))
        
        L = Concatenate(axis=1)([Le, La])
        L = Conv3D(output_channels, 1, padding="same", use_bias=False, data_format=self.DT)(L)
        L = BatchNormalization(axis=1)(L)
        L = Lambda(self._swish, output_shape=K.int_shape(L)[1:])(L)
            
        return L
    

    def SSLR(self, x, n_filters):
        '''
        spectral-spatial local representation module
        '''        
        L = self.inverted_residual_block(x, n_filters[0], n_filters[1], (3, 3, 7))
        
        return L


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
    
    
    # centralized dot-product attention
    def CDPA(self, q, r, v, use_centralization=True):
        n_head, n_row, n_col, n_band = K.int_shape(q)[1:]
        
        # dot-product between 5-d tensors
        def _matmul(x):            
            n_head, n_row, n_col, n_band = K.int_shape(x[0])[1:]
            
            x1 = Reshape((n_head, n_row * n_col, n_band))(x[0])
            x2 = Reshape((n_head, n_row * n_col, n_band))(x[1])
            x2 = Permute((1, 3, 2))(x2)
            
            m = tf.matmul(x1, x2)
            m = Reshape((n_head, n_row * n_col, n_row * n_col, 1))(m)
            
            return m
        
        U = Lambda(_matmul, output_shape=(n_head, n_row * n_col, n_row * n_col, 1))([q, r])

        if use_centralization:        
            # centralization
            C = self.centralization(U)
        else:
            C = U
        
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
        A = Lambda(_matmul2, output_shape=(n_head, n_row, n_col, n_band))([w, v])
        A = Reshape((n_head, n_row, n_col, n_band))(A)
        
        return A

    
    def CMHA(self, x, rlv, n_head, kernel_size=(3,3,7), use_ARE=True, use_centralization=True):
        '''
        centralized multi-head self-attention module
        input: x, rlv(relevance)
        output: A
        '''
        n_channel, n_row, n_col, n_band = K.int_shape(x)[1:]
        
        # relevancy
        def _repeat(x):
            return K.tile(x, (1, 1, 1, 1, 1))
        
        def _multiply(x):
            return x[0] * x[1]
        
        if use_ARE:
            rlv = Lambda(_repeat, output_shape=(1, n_row, n_col, 1))(rlv)
            _R = Lambda(_multiply, output_shape=(n_channel, n_row, n_col, n_band))([x, rlv])
        else:
            _R = x
            
        # linear layers
        Q = Conv3D(filters=n_head, kernel_size=1, strides=1, padding="same", data_format=self.DT)(x)
        R = Conv3D(filters=n_head, kernel_size=1, strides=1, padding="same", data_format=self.DT)(_R)
        V = Conv3D(filters=n_head, kernel_size=1, strides=1, padding="same", data_format=self.DT)(x)
        
        # CDPA module
        A = self.CDPA(Q, R, V, use_centralization=use_centralization)
        
        # linear layer
        A = Conv3D(filters=n_head, kernel_size=1, strides=1, data_format=self.DT)(A)
        A = BatchNormalization(axis=1)(A)
        A = Lambda(self._swish, output_shape=K.int_shape(A)[1:])(A)
        
        return A


    def spectral_relevance(self, x):
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
        spe_relevance = Lambda(_sim_cos, output_shape=(1,n_row,n_col,1))(cx_c_x)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        # linear transformation (alignment)
        # [-1, 1] --> [0, 1]
        def _alignment(x, a1=-1., b1=1., a2=0., b2=1.):
            return ((x - a1) / (b1 - a1)) * (b2 - a2) + a2
        
        spe_relevance = Lambda(_alignment, output_shape=(1,n_row,n_col,1))(spe_relevance)
        
        return spe_relevance
    
    
    def spatial_relevance(self, x, I_gaussian):
        '''
        value range: [0, 1]
        '''
        n_row = K.int_shape(x)[-3]
        
        # offset and reorder
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
        
        spa_relevance = Lambda(_gaussian_2d, output_shape=(1, n_row, n_row, 1),
                               arguments={"I_gaussian": I_gaussian})(x)
            
        return spa_relevance
    
    
    def _neutral_fuse(self, x):
        return (x[0] + x[1]) / 2.
      

    def ARE(self, x, I_gaussian):
        '''
        Adaptive Relevance Estimation
        input:  x, I_gaussian
        output: relevance mask
        '''
        n_row, n_col = K.int_shape(x)[-3:-1]
        
        # relevance
        spe_rlv = self.spectral_relevancy(x)
        spa_rlv = self.spatial_relevancy(x, I_gaussian)
        
        # adaptive fuse
        rlv = Concatenate(axis=-1)([spe_rlv, spa_rlv])
        relevance = Adaptive_Sum(output_dim=(1, n_row, n_col, 1))(rlv)
        
        return relevance

   
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
                          pool_size=(pool_size, pool_size, 1),
                          strides=(strides, strides, 1),
                          padding="valid",
                          data_format=self.DT,
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 1, n_band)
            r2 = K.pool3d(x[:, :, 0:center[0], center[1]:center[1] + 1, :],
                          pool_size=(pool_size, 1, 1),
                          strides=(strides, 1, 1),
                          padding="valid",
                          data_format=self.DT,
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 4, n_band)
            r3 = K.pool3d(x[:, :, 0:center[0], center[1] + 1:, :],
                          pool_size=(pool_size, pool_size, 1),
                          strides=(strides, strides, 1),
                          padding="valid",
                          data_format=self.DT,
                          pool_mode="avg")
            
            # (bs, n_channle, 1, 4, n_band)
            r8 = K.pool3d(x[:, :, center[0]:center[0] + 1, 0:center[1], :],
                          pool_size=(1, pool_size, 1),
                          strides=(1, strides, 1),
                          padding="valid",
                          data_format=self.DT,
                          pool_mode="avg")
            
            # (bs, n_channle, 1, 1, n_band)
            r0 = x[:, :, center[0]:center[0] + 1, center[1]:center[1] + 1, :]
            
            # (bs, n_channle, 1, 4, n_band)
            r4 = K.pool3d(x[:, :, center[0]:center[0] + 1, center[1] + 1:, :],
                          pool_size=(1, pool_size, 1),
                          strides=(1, strides, 1),
                          padding="valid",
                          data_format=self.DT,
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 4, n_band)
            r7 = K.pool3d(x[:, :, center[0] + 1:, 0:center[1], :],
                          pool_size=(pool_size, pool_size, 1),
                          strides=(strides, strides, 1),
                          padding="valid",
                          data_format=self.DT,
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 1, n_band)
            r6 = K.pool3d(x[:, :, center[0] + 1:, center[1]:center[1] + 1, :],
                          pool_size=(pool_size, 1, 1),
                          strides=(strides, 1, 1),
                          padding="valid",
                          data_format=self.DT,
                          pool_mode="avg")
            
            # (bs, n_channle, 4, 4, n_band)
            r5 = K.pool3d(x[:, :, center[0] + 1:, center[1] + 1:, :],
                          pool_size=(pool_size, pool_size, 1),
                          strides=(strides, strides, 1),
                          padding="valid",
                          data_format=self.DT,
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
                          data_format=self.DT,
                          pool_mode="avg")
        
        print("pool_size:", pool_size, ", strides:", strides, ", pool_band:", pool_band, ", strids_band:", strides_band)
        print("after CP, shape is", K.int_shape(output), "\n")
        
        return output
        
        
    def encoder(self, x, rlv, n_head, n_filters, i, use_ARE=True, use_SSLR=True, use_CP=True, use_centralization=True):
        if K.int_shape(x)[1] != n_head:
            x = Conv3D(n_head, 1, data_format=self.DT, use_bias=False)(x)
        
        if use_SSLR:
            # SSLR module
            L = self.SSLR(x, n_filters)
        else:
            L = x
            
        # CMHA module
        A = self.CMHA(L, rlv, n_head, use_ARE=use_ARE, use_centralization=use_centralization)
        
        if use_SSLR:
            # FFP
            F = Concatenate(axis=1)([A, L])
            F = Conv3D(n_head, 1, data_format=self.DT, use_bias=False)(F)
            F = Lambda(self._swish, output_shape=K.int_shape(F)[1:])(F)
        else:
            pass
        
        F = Add()([F, x])
        
        shape = K.int_shape(F)
        if use_CP:
            print("CP Layer")
            new_shape = (shape[1],) + self.CP_shapes[i]
            F = Lambda(self._circular_pooling, output_shape=new_shape,
                       arguments={"pool_size": 2, "strides": 1})(F)
        else:
            if shape[-2] >= 2:
                F = MaxPooling3D(pool_size=2, strides=2, data_format=self.DT, padding="same")
            else:
                F = MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2), data_format=self.DT, padding="same")
                      
        return F
    
    
    def CERF(self, m1, m2_, i):
        '''
        cross-encoder relevancy fusion (CERF) module
        '''
        n_row_1 = K.int_shape(m1)[2]
        n_row_2 = K.int_shape(m2_)[2]
        
        print("CERF module")
        m1 = Lambda(self._circular_pooling, output_shape=(1,) + self.CP_shapes[i],
                    arguments={"pool_size": self.cp_pool_size, "strides": self.cp_strides,
                               "pool_band": 1, "strides_band": 1})(m1)

        def _concat(x):
            return tf.concat(x, axis=-1)
        m = Lambda(_concat, output_shape=(1, n_row_2, n_row_2, 1))([m2_, m1])
        m = Adaptive_Sum(output_dim=(1, n_row_2, n_row_2, 1))(m)
        
        return m


    def build_model(self):
        # input
        I = Input(shape=self.input_shape)
        
        # an input puppet used to provide the width of input and generate the Gaussian 2-d distribution map
        # this is limited by keras and tensorflow frameworks
        shape_gaussian = (1, self.input_shape[2], 1, 1)
        I_gaussian = Input(shape=shape_gaussian)
        
        # conv embedding
        embedding = Conv3D(filters=1, kernel_size=(1, 1, 3), padding="same",
                            use_bias=False, data_format=self.DT)(I)
        
        i = 0
        print(1)
        # ARE1
        M1 = self.ARE(embedding, I_gaussian)
        # encoder1
        F1 = self.encoder(embedding, M1, self.n_head[i], self.n_filter[i], i)
        
        i += 1
        print(2)
        # ARE2
        M2_ = self.ARE(F1, I_gaussian)
        # CERF
        M2 = self.CP_RF(M1, M2_, i)
        # encoder2
        F2 = self.encoder(F1, M2, self.n_head[i], self.n_filter[i], i)
        
        i += 1
        print(3)
        # ARE3
        M3_ = self.ARE(F2, I_gaussian)
        # CERF
        M3 = self.CP_RF(M2, M3_, i)
        # encoder3
        F3 = self.encoder(F2, M3, self.n_head[i], self.n_filter[i], i)
        
        # Cls
        F = Flatten()(F3)
        F = Dropout(0.5)(F)
        y = Dense(self.n_category, activation="softmax")(F)
        
        self.model = Model(inputs=[I, I_gaussian], outputs=y, name="centralformer")
        


# ==================================================================================
# ================================training==========================================
#===================================================================================
if __name__ == "__main__":
    # optimizer
    lr = 0.001
    rmsprop = RMSprop(learning_rate=lr)

    from tensorflow.keras.callbacks import LearningRateScheduler
    # callback function
    def decay_schedule(epoch, learning_rate):
        if epoch % 40 == 0 and epoch != 0:
            learning_rate = learning_rate * 0.5
        return learning_rate
    lr_scheduler = LearningRateScheduler(decay_schedule)

    # prepare your training, validation, and test samples here (random seed: 42)
    # shapes: Indian Pines (1, 11, 11, 200), Pavia University: (1, 5, 5, 103), Loukia: (1, 7, 7, 176), Dioni: (1, 5, 5, 176)
    input_shape = [1, 11, 11, 200]

    # all sample
    X = None    # (bs, 1, 11, 11, 200)
    y = None    # (bs, 1)
    # training samples
    X_train = X[index]      # (bs_training, 1, 11, 11, 200) 
    y_train = None          # (bs_training, 1)
    y_train_1hot = None     # (bs_training, 16)
    # validation samples    
    X_val = None
    y_val = None
    y_val_1hot = None
    # test samples
    X_test = None
    y_test = None
    y_test_1hot = None

    # you can prepare the input puppet (I_gaussian) here
    puppet = []
    for i in range(args.width):
        puppet.append(i)
    puppet = np.array(puppet, dtype=np.float32)                     # (n,)
    puppet = np.expand_dims(puppet, axis=(0, 1, 3, 4))              # (1, 1, n, 1, 1)

    puppet_training = np.repeat(puppet, X_train.shape[0], axis=0)   # (bs_training, 1, n, 1, 1)
    puppet_val = np.repeat(puppet, X_val.shape[0], axis=0)          # (bs_val, 1, n, 1, 1)
    puppet_test = np.repeat(puppet, X_test.shape[0], axis=0)        # (bs_test, 1, n, 1, 1)

    # build model
    network = CentralFormer(input_shape=input_shape, n_category=args.n_category)
    model = network.model

    # compile
    model.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())

    # training
    best_OA = 0.
    best_state = None
    time_training = 0.
    for e in range(args.exp):
        for i in range(args.epoch):
            print("Exp: {0}/{1}, Epoch: {2}/{3}".format(e + 1, args.exp, i + 1, args.epochs))
            t1 = time.time()
            hist = model.fit(x=[X_train, puppet_train], y=y_train_1hot, batch_size=args.bs, epochs=1, shuffle=True, validation_data=([X_val, puppet_val], y_val_1hot), verbose=2)
            t2 = time.time()
            time_training += (t2 - t1)

            if hist.history[key_val][0] >= best_OA:
                best_OA = hist.history[key_val][0]
                best_state = model.get_weights()
                print("best OA: ", best_OA)
            
            if (i+1) % 40 == 0:
                lr*=0.5
                rmsprop = RMSprop(learning_rate=lr)
                model.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=["accuracy"])

    # test
    model.evaluate(x=[X_test, puppet_test], y=y_test_1hot)
    y_pred = model.predict([X_test, puppet_test])



