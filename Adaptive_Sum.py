# -*- coding: utf-8 -*-

from keras import initializers, constraints, regularizers 
from keras.engine.base_layer import Layer
from keras import backend as K
from keras.utils.generic_utils import to_list


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
        
        