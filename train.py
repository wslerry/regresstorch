import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from utils import Dataset
import numpy as np


def selection(data):
    namelist = list()
    for x in data.columns.values:
        namelist.append(x)

    return namelist


def train():
    input_dir, var1, var2 = opt.input, opt.var1, opt.var2
    data = Dataset(input_dir).open()
    colnames = selection(data)
    print(colnames)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./example/data.csv', help='*.csv path')
    parser.add_argument('--var1', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--var2', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    # train()

    with torch.no_grad():
        train()
