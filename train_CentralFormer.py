# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.optimizers import RMSprop
import time
import winsound as ws
import tensorflow as tf
from keras.utils import np_utils
import os

# customized packages
from Parameter import args
from CentralFormer import CentralFormer
import utils


project_folder = os.getcwd()        # your own project folder name, such as "D:\\project\\"
ds_name = args.ds_name.lower()      # lower all capitalized letter of the name of data set

# Prepare your own training, validation, and test samples here
# You can also download our processed samples here (https://drive.google.com/drive/folders/1Htr4jgtJyRT24VSbVbg2jED7kXAYUGqV?usp=drive_link).
# We generated these samples with the random seed 42.
# Shapes: Indian Pines (1, 11, 11, 200), Pavia University: (1, 5, 5, 103), Loukia: (1, 7, 7, 176), Dioni: (1, 5, 5, 176)
# training : validation : test => Indian Pines (0.05 : 0.05: 0.9), Pavia University: (0.02: 0.05: 0.93), Loukia: (0.05 : 0.05: 0.9), Dioni: (0.05 : 0.05: 0.9)

# taking Indian Pines data set as an example
input_shape = [1, args.width[args.ds_name], args.width[args.ds_name], args.band[args.ds_name]]

X_train = np.load(os.path.join(project_folder, "CentralFomer", "samples", ds_name + "_x_training.npy"))   # (bs_training, 1, 11, 11, 200) 
y_train = np.load(os.path.join(project_folder, "CentralFomer", "samples", ds_name + "_y_training.npy"))   # (bs_training, 1)
# remember to convert label to the one-hot form
y_train_1hot = np_utils.to_categorical(y_train, args.n_category[args.ds_name])                            # (bs_training, 16)

# validation samples
X_val = np.load(os.path.join(project_folder, "CentralFomer", "samples", ds_name + "_x_val.npy"))
y_val = np.load(os.path.join(project_folder, "CentralFomer", "samples", ds_name + "_y_val.npy"))
y_val_1hot = np_utils.to_categorical(y_val, args.n_category[args.ds_name])

# test samples
X_test = np.load(os.path.join(project_folder, "CentralFomer", "samples", ds_name + "_x_test.npy"))
y_test = np.load(os.path.join(project_folder, "CentralFomer", "samples", ds_name + "_y_test.npy"))
y_test_1hot = np_utils.to_categorical(y_test, args.n_category[args.ds_name])


# The input puppet (I_gaussian) can be acquired (https://drive.google.com/drive/folders/1Htr4jgtJyRT24VSbVbg2jED7kXAYUGqV?usp=drive_link) here.
# Or you can also generate different widths of puppets with the code below.
puppet = []
for i in range(args.width[args.ds_name]):
    puppet.append(i)
puppet = np.array(puppet, dtype=np.float32)                     # (n,)
puppet = np.expand_dims(puppet, axis=(0, 1, 3, 4))              # (1, 1, n, 1, 1)

puppet_train = np.repeat(puppet, X_train.shape[0], axis=0)      # (bs_training, 1, n, 1, 1)
puppet_val = np.repeat(puppet, X_val.shape[0], axis=0)          # (bs_val, 1, n, 1, 1)
puppet_test = np.repeat(puppet, X_test.shape[0], axis=0)        # (bs_test, 1, n, 1, 1)

# If you have downloaded puppet files, just load them with np.load() method.
# puppet_train = np.load(os.path.join(project_folder, "CentralFomer", "samples", ds_name + "_puppet_training.npy"))
# puppet_val = np.load(os.path.join(project_folder, "CentralFomer", "samples", ds_name + "_puppet_val.npy"))
# puppet_test = np.load(os.path.join(project_folder, "CentralFomer", "samples", ds_name + "_puppet_test.npy"))


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

if tf.__version__ == "2.0.0":
    key_val = "val_accuracy"
else:
    key_val = "val_acc"


# build model
network = CentralFormer(input_shape=input_shape, n_category=args.n_category[args.ds_name])
model = network.model

# compile
model.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=["accuracy"])

# load weights with (1)keras package or (2)the methods of network class 
# 1. model.load_weights(r"D:\...\CentralFormer\weights\indian_pines.h5")
# 2. network.load_weights(r"D:\...\CentralFormer\weights\indian_pines.pickle")


# training
best_OA = 0.
best_state = None
time_training = 0.

for i in range(args.epochs):
    print("\nEpoch: {0}/{1}, lr: {2}, best_OA: {3}".format(i + 1, args.epochs, lr, best_OA))
    t1 = time.time()
    hist = model.fit(x=[X_train, puppet_train], y=y_train_1hot, batch_size=args.bs, epochs=1, shuffle=True, validation_data=([X_val, puppet_val], y_val_1hot), verbose=1)
    t2 = time.time()
    time_training += (t2 - t1)

    if hist.history[key_val][0] >= best_OA:
        best_OA = hist.history[key_val][0]
        best_state = model.get_weights()
        model.set_weights(best_state)
        print("best OA: ", best_OA)
    
    if (i + 1) % 40 == 0 and i < args.epochs - 1:
        lr *= 0.5
        rmsprop = RMSprop(learning_rate=lr)
        model.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=["accuracy"])
        model.set_weights(best_state)

print("best OA: ", best_OA)
model.set_weights(best_state)
model.save_weights(np.load(os.path.join(project_folder, "CentralFomer", "weights", ds_name + ".h5")))
network.save_weights(np.load(os.path.join(project_folder, "CentralFomer", "weights", ds_name + ".pickle")))

# play sound
if args.env == 0:
    ws.PlaySound("C:\\Windows\\Media\\Alarm02.wav", ws.SND_ASYNC)

# test
model.evaluate(x=[X_test, puppet_test], y=y_test_1hot)

# test
print("testing ...\n")
t1_ = time.time()
y_pred = model.predict([X_test, puppet_test])
t2_ = time.time()
time_test = t2_ - t1_


# quantitative evaluation
metrics = utils.compute_metrics(y_test_1hot, y_pred, args.n_category[args.ds_name])
matrix = utils.confusion_matrix(y_test_1hot, y_pred, args.n_category[args.ds_name])
kappa = utils.kappa(matrix)

print(metrics, matrix, kappa, time_training, time_test)


# qualitative evaluation (predict all labeled pixels)
X = np.load(os.path.join(project_folder, "CentralFomer", "samples", "4prediction", ds_name + ".npy"))
pos = np.load(os.path.join(project_folder, "CentralFomer", "samples", "4prediction", ds_name + "_pos.npy"))
puppet_all = np.repeat(puppet, X.shape[0], axis=0)   # (bs_all_labeled_pixels, 1, n, 1, 1)
y = model.predict([X, puppet_all])

predicted_map = utils.get_predicted_map(np.argmax(y, axis=1), pos, args.row[args.ds_name], args.col[args.ds_name])
utils.save_predicted_map(predicted_map, os.path.join(project_folder, "CentralFomer", "samples", "results", ds_name + ".png"))


  
# print finish time
print("\n", time.ctime(time.time()))

