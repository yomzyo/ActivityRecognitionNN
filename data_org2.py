import numpy as np


class data_org2():
    def __init__(self, dataset='ntu_rgbd_dataset'):
        self.dataset = dataset
        self.partition = np.load('partition.npy')
        self.labels = np.load('labels.npy')
        self.label_ids = np.load('label_ids.npy')
