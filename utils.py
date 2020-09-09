import os
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

class Dataset:
    def __init__(self):
        super(Dataset, self).__init__()
        # self.directory = None

    def open(self,directory):
        dirs = self.directory
        return pd.read_csv(dirs)


def select_device(device='',batch_size=None):
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device

    cuda = False if cpu_request else torch.cuda.is_available()

    if cuda:
        c=1024 ** 2
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA'
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print(f"{s}sdevice{i} _CudaDeviceProperties(name='{x[i].name}', total_memory={x[i].total_memory / c}MB)")
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')
    
