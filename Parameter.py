# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 10:12:35 2022

@author: fes_map
"""

import argparse


parser = argparse.ArgumentParser(description="Networks")

parser.add_argument("--env", type=int, default=0, help="type of operation system (0: windows, 1: ubuntu)")
parser.add_argument("--ds_name", type=str, default="Indian_Pines", help="name of data set")

parser.add_argument("--train_ratio", type=dict, default={"Indian_Pines": 0.05, "PaviaU": 0.02, "Loukia": 0.05, "Dioni": 0.05}, help="training sample proportion")
parser.add_argument("--val_ratio", type=dict, default={"Indian_Pines": 0.05, "PaviaU": 0.05, "Loukia": 0.05, "Dioni": 0.05}, help="training sample proportion")
parser.add_argument("--width", type=dict, default={"Indian_Pines": 11, "PaviaU": 5, "Loukia": 7, "Dioni": 5}, help="width of sample")
parser.add_argument("--n_category", type=dict, default={"Indian_Pines": 16, "PaviaU": 9, "Loukia": 14, "Dioni": 12}, help="number of category")
parser.add_argument("--band", type=dict, default={"Indian_Pines": 200, "PaviaU": 103, "Loukia": 176, "Dioni": 176}, help="number of band")
parser.add_argument("--row", type=dict, default={"Indian_Pines": 145, "PaviaU": 610, "Loukia": 249, "Dioni": 250}, help="number of row")
parser.add_argument("--col", type=dict, default={"Indian_Pines": 145, "PaviaU": 340, "Loukia": 945, "Dioni": 1376}, help="number of col")

parser.add_argument("--bs", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=200, help="number of iteration")
parser.add_argument("--exp", type=int, default=10, help="number of experiments")
parser.add_argument("--stack", type=int, default=3, help="number of residual blocks")

parser.add_argument("--cp_pool_size", type=int, default=2, help="pool_size of CP layer")
parser.add_argument("--cp_strides", type=int, default=1, help="strides of CP layer")
parser.add_argument("--cp_pool_band", type=int, default=2, help="pool_size of CP layer in spectral dimension")
parser.add_argument("--cp_strides_band", type=int, default=2, help="strides of CP layer in spectral dimension")

args = parser.parse_args()
args.ds_name = "Indian_Pines"





