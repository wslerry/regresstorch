import argparse
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import utils
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
    x, y = X_train, y_train

    return x, y


def train():
    input_dir, var1, var2, adam, device = opt.input, opt.var1, opt.var2, opt.adam, opt.device

    data = Dataset(input_dir).open()

    device = utils.select_device(device, batch_size=opt.batch_size)

    for i in (var1, var2):
        variable_check(data, i)

    use_cuda = torch.cuda.is_available()

    X_reshape = data[var1].values.reshape(-1, 1)
    y_reshape = data[var2].values.reshape(-1, 1)

    if use_cuda:
        x, y = gpu_dataset(X_reshape, y_reshape)
    else:
        x, y = cpu_dataset(X_reshape, y_reshape)

    # Initialize model
    net = LinearNet(n_feature=x.size(1), n_output=y.size(1)).to(device)

    if adam:
        # optimizer using Adam
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    else:
        # optimizer using SGD
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    loss_func = nn.MSELoss()

    batch_size = opt.batch_size
    n_epochs = opt.epoch
    batch_no = len(x) // batch_size
    train_loss = 0
    train_loss_min = np.Inf
    
    if use_cuda:
        for epoch in range(n_epochs):
            for i in range(batch_no):
                start = i* batch_size
                end = start + batch_size

                optimizer.zero_grad()
                prediction = net(x)
                loss = loss_func(prediction, y)
                loss.backward()
                optimizer.step()
                values, labels = torch.max(prediction, 1)
                num_right = np.sum(labels.cpu().data.numpy() == y[start:end])
                train_loss += loss.item()*batch_size

            train_loss = train_loss / len(x)
            if train_loss <= train_loss_min:
                print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min,train_loss))
                torch.save(net.state_dict(), "regression_networks.pt")
                train_loss_min = train_loss

            if epoch % 50 == 0:
                print('')
                print("Epoch: {} \tTrain Loss: {} \tTrain Accuracy: {}".format(epoch+1, train_loss,num_right / len(y[start:end]) ))
        print('Training Ended! ')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help='epoch value')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--input', type=str, default='./example/data.csv', help='*.csv path')
    parser.add_argument('--var1', type=str, default='H', help='independent variable')
    parser.add_argument('--var2', type=str, default='VUB', help='dependent variable')
    parser.add_argument('--adam', action='store_true', default=True, help='use adam optimizer')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    print(opt)

    # device = utils.select_device(opt.device, batch_size=opt.batch_size)
    # print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # tb_writer = SummaryWriter(comment=opt.name)

    train()

