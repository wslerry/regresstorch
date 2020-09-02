import pandas as pd


class Dataset:
    def __init__(self, directory):
        super(Dataset, self).__init__()
        self.directory = directory

    def open(self):
        x = pd.read_csv(self.directory)
        return x
