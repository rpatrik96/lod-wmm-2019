import pickle
from os.path import abspath, dirname, join, splitext, isfile

import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms

"""-------------------------------------------------------------------------"""
"""--------------------------Sin+Sin-Noise----------------------------------"""
"""-------------------------------------------------------------------------"""
time_series_window_size = 50

class TimeSeriesData(Dataset):
    def __init__(self, train=True, window_size=time_series_window_size, num_test=10000, noise=False):
        self.num_train_valid = 60000
        self.num_test = num_test
        self.train = train
        self.window_size = window_size
        self.noise = noise

        num_elem = self.num_train_valid + self.num_test + 2 * self.window_size
        lp = np.linspace(-num_elem / 100 * np.pi, num_elem / 100 * np.pi, num_elem)
        data0 = np.sin(lp) * 3 + np.cos(lp * 2) + np.sin(np.pi / 2 + lp * 4) * 1.5 + np.log(np.abs(lp))
        data1 = np.cos(lp) + 2 + np.cos(np.exp(lp // 10 - 5))
        data2 = np.sin(lp * np.cos(lp)) + 2 * np.cos(5 * (lp - 2) + np.pi / 2)

        # data needs to be 3d (batch_size, window_size, num_features)
        data0 = np.float32(data0)
        data1 = np.float32(data1)
        data2 = np.float32(data2)

        data = np.stack((data0, data1, data2)).transpose()

        self.train_valid_dataset = data[:self.num_train_valid + self.window_size]
        self.test_dataset = data[-(self.num_test + self.window_size):]

        if self.noise:
            n = np.eye(3, 3) + np.random.rand(3, 3)
            self.train_valid_dataset_noised = np.float32(np.matmul(self.train_valid_dataset, n))
            self.test_dataset_noised = np.float32(np.matmul(self.test_dataset, n))

    def __len__(self):

        if self.train:
            return self.num_train_valid
        else:
            return self.num_test

    def __getitem__(self, idx):

        if self.train:
            data = self.train_valid_dataset
            target = self.train_valid_dataset if not self.noise else self.train_valid_dataset_noised
        else:
            data = self.test_dataset
            target = self.test_dataset if not self.noise else self.test_dataset_noised

        return [
            data[idx: idx + self.window_size],
            target[idx + self.window_size]
            ]

"""-------------------------------------------------------------------------"""
"""-------------------------------MIDIData----------------------------------"""
"""-------------------------------------------------------------------------"""

class MIDIData(Dataset):
    def __init__(self, dataset, train=False, validate=False, test=False, window_size=2):
        self.train = train
        self.validate = validate
        self.test = test
        self.window_size = min(25, window_size)  # 25 is minimum sequence length
        self.transform = transforms.Compose([transforms.ToTensor()])

        if dataset == 'jsb-chorales':
            datadir = join(join(join(dirname(dirname(abspath(__file__))), 'datasets'), 'jsb-chorales'), 'data')
            data = loadmat(join(datadir, 'JSB_Chorales.mat'))
            self.traindata = self._generate_data(data['traindata'][0])
            self.validdata = self._generate_data(data['validdata'][0])
            self.testdata = self._generate_data(data['testdata'][0])
        else:
            raise ValueError("Invalid argument: ", dataset)

    def _convert_pickle(self, dataset, pickle_file):
        datadir = join(join(join(dirname(dirname(abspath(__file__))), 'datasets'), dataset), 'data')
        filename = splitext(pickle_file)[0] + '.npy'

        # if converted, just load
        if isfile(join(datadir, filename)):
            data = np.load(join(datadir, filename)).item()
        else:
            data_pickle = pickle.load(open(join(datadir, pickle_file), 'rb'))
            data = {}
            data['train'] = self._to_one_hot(data_pickle['train'])
            data['valid'] = self._to_one_hot(data_pickle['valid'])
            data['test'] = self._to_one_hot(data_pickle['test'])
            np.save(join(datadir, filename), data)

        return data

    def _to_one_hot(self, data):
        midi_offset = 21  # midi keys are numbered from 21 to 108
        num_classes = 88  # number of piano keys
        one_hot_list = []

        for item in data:
            encoded_list = np.zeros((len(item), num_classes))

            # convert each step to one-hot
            for idx, elem in enumerate(item):
                encoded_list[idx][np.asarray(elem) - midi_offset if len(elem) else elem] = 1
            one_hot_list.append(encoded_list)

        return one_hot_list

    def __len__(self):

        if self.train:
            return len(self.traindata)
        elif self.validate:
            return len(self.validdata)
        elif self.test:
            return len(self.testdata)
        else:
            raise ValueError("No appropriate flag from {test, validate, train} is specified!")

    def __getitem__(self, idx):

        if self.train:
            data = self.traindata
        elif self.validate:
            data = self.validdata
        elif self.test:
            data = self.testdata

        return data[idx][:-1], data[idx][-1]

    def _generate_data(self, data):

        container = []
        # sequences
        for seq in data:
            # windows
            for i in range(len(seq) - self.window_size):
                container.append(np.float32(seq[i:i + self.window_size + 1]))

        return container
