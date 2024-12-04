# -*- coding: utf-8 -*-

import pickle


class Network():
    '''
    network baseclass
    '''
    def __init__(self, name, input_shape=(1, 1, 1), n_category=1, filters=8, kernel_size=(3, 3, 1)):
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
        
        self.input_shape, self.n_category, self.filters, self.kernel_size = input_shape, n_category, filters, kernel_size
        self.DT=self.data_format = "channels_first"
        
        # print("\nFrom basic class 'Network':\nArguments 'kernel_size' of all inherited models denote the spatial size of the convolutional kernels used to extract the spatial features.\n")
        
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
        
        