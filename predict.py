import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from utils import Dataset


def predict():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./example/data.csv', help='*.csv path')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        predict()
