from os.path import abspath, dirname, isfile, join
from time import gmtime, strftime

import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from datasets import TimeSeriesData, MIDIData

"""-------------------------------------------------------------------------"""
"""----------------------------Dataset descriptor---------------------------"""
"""-------------------------------------------------------------------------"""

class DataSetDescriptor(object):
    def __init__(self, device, loss_fn, dataset='mnist', model=None, num_valid=5000, num_seeds=1, generate=False,
                 visualize=False):
        self.device = device
        self.loss_fn = loss_fn

        self.dataset = dataset
        self.num_seeds = num_seeds
        self.num_valid = num_valid

        self.timestamp = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        self.idx_filename = "indices_" + self.dataset + ".hdf5"
        self.base_dir = join(join(dirname(dirname(abspath(__file__))), 'datasets'), self.dataset)

        self._has_separate_datasets()

        # Select dataset
        if not self.has_separate_datasets:
            if "sin" in self.dataset:
                self.problem_type = "regression"

                kwargs = {"noise": True} if "noise" in self.dataset else {}

                self.train_valid_dataset = TimeSeriesData(train=True, **kwargs)
                self.test_dataset = TimeSeriesData(train=False, **kwargs)

            elif self.dataset == 'mnist':
                self.problem_type = "classification"
                # Compose transform
                self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

                # Load train, validation and test sets
                self.train_valid_dataset = datasets.MNIST(join(self.base_dir, 'data'), train=True, download=True,
                                                          transform=self.transform)
                self.test_dataset = datasets.MNIST(join(self.base_dir, 'data'), train=False, transform=self.transform)

            elif self.dataset == 'cifar10':
                self.problem_type = "classification"
                # Compose transform
                self.transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                              (0.24703233, 0.24348505, 0.26158768))])

                # Load train, validation and test sets
                self.train_valid_dataset = datasets.CIFAR10(root=join(self.base_dir, 'data'), train=True, download=True,
                                                            transform=self.transform)
                self.test_dataset = datasets.CIFAR10(root=join(self.base_dir, 'data'), train=False, download=True,
                                                     transform=self.transform)

            self.num_data, self.num_test = len(self.train_valid_dataset), len(self.test_dataset)
            self.num_train = self.num_data - self.num_valid
            self.indices_train, self.indices_test = list(range(self.num_data)), list(range(self.num_test))

            self.train_idx_list, self.validation_idx_list, self.test_idx_list = \
                self._generate_indices() if generate or not isfile(
                        join(self.base_dir, self.idx_filename)) else self._read_indices()

        else:
            if self.dataset == 'jsb-chorales':
                self.problem_type = "multiclass-multilabel"

                # load datasets
                self.train_dataset = MIDIData(dataset=dataset, train=True)
                self.validation_dataset = MIDIData(dataset=dataset, validate=True)
                self.test_dataset = MIDIData(dataset=dataset, test=True)

            # due to compatibility (before more index sets for different random seeds was used)
            self.train_idx_list, self.validation_idx_list, self.test_idx_list = [], [], []

            # shuffle indices
            train_list = list(range(len(self.train_dataset)))
            valid_list = list(range(len(self.validation_dataset)))
            test_list = list(range(len(self.test_dataset)))

            np.random.shuffle(train_list)
            np.random.shuffle(valid_list)
            np.random.shuffle(test_list)

            # assign index lists
            self.train_idx_list.append(train_list)
            self.validation_idx_list.append(valid_list)
            self.test_idx_list.append(test_list)

    def _generate_indices(self):
        train_idx_list, validation_idx_list, test_idx_list = [], [], []

        for i in range(self.num_seeds):
            # shuffle
            np.random.shuffle(self.indices_train)
            np.random.shuffle(self.indices_test)

            # append
            train_idx_list.append(self.indices_train[:self.num_train])
            validation_idx_list.append(self.indices_train[self.num_train:])
            test_idx_list.append(self.indices_test)

        # create file handle for the calibration results in hdf5 format
        with h5py.File(join(self.base_dir, self.idx_filename), 'w') as f:
            # rectification maps for camera0
            f.create_dataset("train_idx_list", data=train_idx_list)
            f.create_dataset("validation_idx_list", data=validation_idx_list)
            f.create_dataset("test_idx_list", data=test_idx_list)

            return train_idx_list, validation_idx_list, test_idx_list

    def _read_indices(self):
        """ read in parameters
         [()] is needed to read in the whole array if you don't do that,
          it doesn't read the whole data but instead gives you lazy access to sub-parts
          (very useful when the array is huge but you only need a small part of it).
         https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py"""
        with h5py.File(join(self.base_dir, self.idx_filename), 'r') as f:
            train_idx_list = f['train_idx_list'][()]
            validation_idx_list = f['validation_idx_list'][()]
            test_idx_list = f['test_idx_list'][()]

        return train_idx_list, validation_idx_list, test_idx_list

    def _has_separate_datasets(self):
        self.has_separate_datasets = False

        if self.dataset == 'jsb-chorales' or self.dataset == 'timit' or self.dataset == 'nottingham' or self.dataset == 'muse':
            self.has_separate_datasets = True

    def generate_datasets(self, seed_idx, batch_size, test_batch_size, device):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device is "cuda" else {}

        # PyTorch does not support DataLoader split,
        # hacked with https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        # Sampler object creation for random shuffling
        valid_sampler = SubsetRandomSampler(self.validation_idx_list[seed_idx])
        train_sampler = SubsetRandomSampler(self.train_idx_list[seed_idx])
        test_sampler = SubsetRandomSampler(self.test_idx_list[seed_idx])

        # DataLoader objects for train, validation and test subsets
        train_loader = DataLoader(
                self.train_valid_dataset if not self.has_separate_datasets else self.train_dataset,
                sampler=train_sampler, batch_size=batch_size, drop_last=False, **kwargs)

        valid_loader = DataLoader(
                self.train_valid_dataset if not self.has_separate_datasets else self.validation_dataset,
                sampler=valid_sampler, batch_size=batch_size, drop_last=False, **kwargs)

        test_loader = DataLoader(self.test_dataset, sampler=test_sampler, batch_size=test_batch_size,
                                 drop_last=False, **kwargs)

        return train_loader, valid_loader, test_loader

"""-------------------------------------------------------------------------"""
"""----------------------------Model parameters-----------------------------"""
"""-------------------------------------------------------------------------"""

class ModelParameters(object):
    def __init__(self, **kwargs):
        # sin + sin-noise + jsb-chorales + timit
        # self.lr = 0.05
        # self.momentum = 0.75
        # self.l2 = 1e-4
        # self.patience = 150

        # sin + sin-noise + jsb-chorales
        self.lr = 0.01
        self.momentum = 0.75
        self.l2 = 1e-4
        self.patience = 150

        # mnist + cifar10
        # self.lr = 0.01
        # self.momentum = 0.9
        # self.l2 = 0.0
        # self.patience = 25

        # common
        self.batch_size = 128  # todo: for JSB only 32
        self.test_batch_size = 512
        self.num_epochs = 200

        self.__dict__ = {**self.__dict__, **kwargs}
