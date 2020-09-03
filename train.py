import argparse
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import Dataset
from model import LinearNet
import matplotlib.pyplot as plt
import imageio
import numpy as np


def selection(data):
    namelist = list()
    for x in data.columns.values:
        namelist.append(x)

    return namelist


def variable_check(data, variable):
    if variable not in selection(data):
        print('[INFO] Missing variable!')
        sys.exit(0)
    else:
        pass
    # index = dict((y, x) for x, y in enumerate(selection(data)))
    # if variable is not None:
    #     try:
    #         var = index[variable]
    #     except KeyError:
    #         print("Variable is empty or not found!")
    #         sys.exit(0)
        # else:
        #     print(f"Variable '{variable}:{var}' is exist.")
        #     pass


def gpu_dataset(X, Y):
    X_tensor = torch.FloatTensor(X).cuda()
    y_tensor = torch.FloatTensor(Y).cuda()
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor,
                                                        test_size=0.2,
                                                        random_state=0)
    x, y = Variable(X_train), Variable(y_train)

    return x, y


def cpu_dataset(X, Y):
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(Y)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor,
                                                        test_size=0.2,
                                                        random_state=0)
    x, y = Variable(X_train), Variable(y_train)

    return x, y

def train():
    input_dir, var1, var2, adam = opt.input, opt.var1, opt.var2, opt.adam

    data = Dataset(input_dir).open()

    for i in (var1, var2):
        variable_check(data, i)

    use_cuda = torch.cuda.is_available()

    X_reshape = data[var1].values.reshape(-1, 1)
    y_reshape = data[var2].values.reshape(-1, 1)

    if use_cuda:
        x, y = gpu_dataset(X_reshape, y_reshape)
    else:
        x, y = cpu_dataset(X_reshape, y_reshape)

    net = LinearNet(n_feature=x.size(1), n_hidden=1000, n_output=y.size(1)).cuda()

    if adam:
        # optimizer using Adam
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    else:
        # optimizer using SGD
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    loss_func = nn.MSELoss()

    # train_losses = []
    #
    # if use_cuda:
    #     for _ in tqdm(range(opt.epoch)):
    #         train_loss = 0
    #         prediction = net(x)
    #         loss = loss_func(prediction, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         train_loss += loss.item()
    #         train_losses.append(loss.item())
    #
    #     print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./example/data.csv', help='*.csv path')
    parser.add_argument('--var1', type=str, default='H', help='independent variable')
    parser.add_argument('--var2', type=str, default='VUB', help='dependent variable')
    parser.add_argument('--epoch', type=int, default=300, help='epoch value')
    parser.add_argument('--adam', action='store_true', default=True, help='use adam optimizer')
    opt = parser.parse_args()
    print(opt)

    # Training(opt.input, opt.var1, opt.var2, opt.epoch, opt.adam)
    # sys.exit(0)

    with torch.no_grad():
        train()
        sys.exit(0)
