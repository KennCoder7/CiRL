# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools
# from tools import *


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        # with open(self.label_path, 'rb') as f:
        #     self.sample_name, self.label = pickle.load(f)
        self.label = np.load(self.label_path)
        # print(self.label[1111])
        self.label = np.squeeze(self.label)
        # print(self.label[1111])
        # exit()

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        data_numpy = self._aug(data_numpy)
        # processing
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

    def _aug(self, data_numpy):
        self.temperal_padding_ratio = 6
        self.shear_amplitude = 0.5
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy


class Feeder_noisy(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 shear_amplitude=0.5, temperal_padding_ratio=6, random_rot_theta=0.3,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.random_rot_theta = random_rot_theta

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            try:
                self.sample_name, self.label_lst, self.label = pickle.load(f)
            except:
                self.label = np.load(self.label_path)
                self.label = np.squeeze(self.label)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        data_numpy = self._aug(data_numpy)
        # processing
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        if self.random_rot_theta > 0:
            data_numpy = tools.random_rot(data_numpy, self.random_rot_theta)

        return data_numpy

class Feeder_noisy_dual(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 shear_amplitude=0.5, temperal_padding_ratio=6, random_rot_theta=0.3,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.random_rot_theta = random_rot_theta

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            try:
                self.sample_name, self.label_lst, self.label = pickle.load(f)
            except:
                self.label = np.load(self.label_path)
                self.label = np.squeeze(self.label)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        data_numpy1 = self._aug(data_numpy)
        data_numpy2 = self._aug(data_numpy)
        # processing
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)

        return [data_numpy1, data_numpy2], label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        if self.random_rot_theta > 0:
            data_numpy = tools.random_rot(data_numpy, self.random_rot_theta)

        return data_numpy

class Feeder_wl(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 n_views=2,
                 bool_contrast=False,
                 shear_amplitude=0.5, temperal_padding_ratio=6, random_rot_theta=0.3,
                 debug=False,
                 mmap=True):
        self.label_index_dict = {}
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.random_rot_theta = random_rot_theta
        self.n_views = n_views
        self.bool_contrast = bool_contrast

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            try:
                self.sample_name, self.label_lst, self.label = pickle.load(f)
            except:
                self.label = np.load(self.label_path)
                self.label = np.squeeze(self.label)
                self.label_lst = None

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

        for i in range(len(self.label)):
            if self.label[i] not in self.label_index_dict:
                self.label_index_dict[self.label[i]] = []
            self.label_index_dict[self.label[i]].append(i)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        if not self.bool_contrast:
            data_numpy = np.array(self.data[index])
            label = self.label[index]
            data_numpy = self._aug(data_numpy)
            if self.label_lst is not None:
                label_lst = np.array(self.label_lst[index])
                return data_numpy, [label, label_lst]
            return data_numpy, label
        # get contrast data
        else:  # (n_views, C, T, V, M)
            data_numpy = [np.array(self.data[index])]
            label = self.label[index]
            instance_label = [np.array(self.label_lst[index])]
            for i in range(1, self.n_views):
                while True:
                    temp_index = random.choice(self.label_index_dict[label])
                    if temp_index != index and self.label[temp_index] == label:
                        break
                data_numpy.append(self._aug(np.array(self.data[temp_index])))
                instance_label.append(np.array(self.label_lst[temp_index]))
            data_numpy = np.array(data_numpy)
            instance_label = np.array(instance_label)
            return data_numpy, label, instance_label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        if self.random_rot_theta > 0:
            data_numpy = tools.random_rot(data_numpy, self.random_rot_theta)

        return data_numpy

class Feeder_wl_dual(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 shear_amplitude=0.5, temperal_padding_ratio=6, random_rot_theta=0.3,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.random_rot_theta = random_rot_theta

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            try:
                self.sample_name, self.label_lst, self.label = pickle.load(f)
            except:
                self.label = np.load(self.label_path)
                self.label = np.squeeze(self.label)
                self.label_lst = None

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        data_numpy1 = self._aug(data_numpy)
        data_numpy2 = self._aug(data_numpy)
        if self.label_lst is not None:
            label_lst = np.array(self.label_lst[index])
            return [data_numpy1, data_numpy2], [label, label_lst]
        # processing
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)
        return [data_numpy1, data_numpy2], label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        if self.random_rot_theta > 0:
            data_numpy = tools.random_rot(data_numpy, self.random_rot_theta)

        return data_numpy


class Feeder_mutual(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 n_views=2,
                 shear_amplitude=0.5, temperal_padding_ratio=6, random_rot_theta=0.3,
                 debug=False,
                 mmap=True):
        self.label_index_dict = {}
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.random_rot_theta = random_rot_theta
        self.n_views = n_views

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            try:
                self.sample_name, self.label_lst, self.label = pickle.load(f)
            except:
                self.label = np.load(self.label_path)
                self.label = np.squeeze(self.label)
                self.label_lst = None

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

        for i in range(len(self.label)):
            if self.label[i] not in self.label_index_dict:
                self.label_index_dict[self.label[i]] = []
            self.label_index_dict[self.label[i]].append(i)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        data_numpy = self._aug(data_numpy)
        if self.label_lst is not None:
            label_lst = np.array(self.label_lst[index])
            return data_numpy, [label, label_lst]
        return data_numpy, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        if self.random_rot_theta > 0:
            data_numpy = tools.random_rot(data_numpy, self.random_rot_theta)
        C, T, V, M = self.C, self.T, self.V, self.M
        mutual_data = np.zeros((C, T, V * 2, 2))
        mutual_data[:, :, :V, 0] = data_numpy[:, :, :, 0]
        mutual_data[:, :, V:, 0] = data_numpy[:, :, :, 1]
        mutual_data[:, :, :V, 1] = data_numpy[:, :, :, 1]
        mutual_data[:, :, V:, 1] = data_numpy[:, :, :, 0]
        return mutual_data


# class Feeder_rwf2000(torch.utils.data.Dataset):
#
#     def __init__(self,
#                  data_path,
#                  label_path,
#                  n_views=2,
#                  bool_contrast=False,
#                  shear_amplitude=0.5, temperal_padding_ratio=6, random_rot_theta=0.3,
#                  debug=False,
#                  mmap=True):
#         self.label_index_dict = {}
#         self.debug = debug
#         self.data_path = data_path
#         self.label_path = label_path
#         self.shear_amplitude = shear_amplitude
#         self.temperal_padding_ratio = temperal_padding_ratio
#         self.random_rot_theta = random_rot_theta
#         self.n_views = n_views
#         self.bool_contrast = bool_contrast
#
#         self.load_data(mmap)
#
#     def load_data(self, mmap):
#         # data: N C V T M
#
#         # load label
#         self.label = np.load(self.label_path)
#
#         # load data
#         if mmap:
#             self.data = np.load(self.data_path, mmap_mode='r')
#         else:
#             self.data = np.load(self.data_path)
#
#         if self.debug:
#             self.label = self.label[0:100]
#             self.data = self.data[0:100]
#             self.sample_name = self.sample_name[0:100]
#
#
#         for i in range(len(self.label)):
#             if self.label[i] not in self.label_index_dict:
#                 self.label_index_dict[self.label[i]] = []
#             self.label_index_dict[self.label[i]].append(i)
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         # get data
#         data_numpy = self.data[index]
#         data_numpy = self.normalize(data_numpy)
#         label = self.label[index]
#         data_numpy = self._aug(data_numpy)
#         if self.label_lst is not None:
#             label_lst = np.array(self.label_lst[index])
#             return data_numpy, [label, label_lst]
#         return data_numpy, label
#
#
#     def _aug(self, data_numpy):
#         if self.temperal_padding_ratio > 0:
#             data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
#
#         if self.shear_amplitude > 0:
#             data_numpy = tools.shear(data_numpy, self.shear_amplitude)
#
#         if self.random_rot_theta > 0:
#             data_numpy = tools.random_rot(data_numpy, self.random_rot_theta)
#
#         return data_numpy
#
#     @staticmethod
#     def normalize(data_numpy):
#         data_numpy = torch.from_numpy(data_numpy)
#         data_numpy = F.normalize(data_numpy, dim=1)
#         data_numpy = data_numpy.numpy()
#         return data_numpy
#
# class Feeder_rwf2000_dual(torch.utils.data.Dataset):
#     def __init__(self,
#                  data_path,
#                  label_path,
#                  shear_amplitude=0.5, temperal_padding_ratio=6, random_rot_theta=0.3,
#                  debug=False,
#                  mmap=True):
#         self.debug = debug
#         self.data_path = data_path
#         self.label_path = label_path
#         self.shear_amplitude = shear_amplitude
#         self.temperal_padding_ratio = temperal_padding_ratio
#         self.random_rot_theta = random_rot_theta
#
#         self.load_data(mmap)
#
#     def load_data(self, mmap):
#         # data: N C V T M
#
#         # load label
#         self.label = np.load(self.label_path)
#
#         # load data
#         if mmap:
#             self.data = np.load(self.data_path, mmap_mode='r')
#         else:
#             self.data = np.load(self.data_path)
#
#         self.N, self.C, self.T, self.V, self.M = self.data.shape
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         # get data
#         data_numpy = self.data[index]
#         data_numpy = self.normalize(data_numpy)
#         label = self.label[index]
#
#         data_numpy1 = self._aug(data_numpy)
#         data_numpy2 = self._aug(data_numpy)
#         # processing
#         # if self.random_choose:
#         #     data_numpy = tools.random_choose(data_numpy, self.window_size)
#         # elif self.window_size > 0:
#         #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
#         # if self.random_move:
#         #     data_numpy = tools.random_move(data_numpy)
#         return [data_numpy1, data_numpy2], label
#
#     def _aug(self, data_numpy):
#         if self.temperal_padding_ratio > 0:
#             data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
#
#         if self.shear_amplitude > 0:
#             data_numpy = tools.shear(data_numpy, self.shear_amplitude)
#
#         if self.random_rot_theta > 0:
#             data_numpy = tools.random_rot(data_numpy, self.random_rot_theta)
#
#         return data_numpy
#
#     @staticmethod
#     def normalize(data_numpy):
#         data_numpy = torch.from_numpy(data_numpy)
#         data_numpy = F.normalize(data_numpy, dim=1)
#         data_numpy = data_numpy.numpy()
#         return data_numpy