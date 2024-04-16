import random
import sys
import os
import math
import pickle5 as pickle
import glob
import copy
import logging
import time
import random
from typing import Any, Callable, Optional, List, Sequence

import multiprocessing
from multiprocessing import Process
from subprocess import call

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import numpy as np
from scipy.io.wavfile import read

import torch
import torch.utils.data
from torch import nn, Tensor
from torch.nn import functional as F
import torch.optim as optim

import torchaudio 
from torchaudio import functional
from torchaudio import transforms

import torchvision
import torchvision.datasets.utils
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation
from torchvision.models._utils import _make_divisible


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from efficientnet_pytorch import EfficientNet

from transformers import HubertForSequenceClassification

import pt_util


class AugmentData():
    def __init__(self, batch_size=64, data_type='train', label_index=0,
                 splits=[16, 48], sub_splits=[10, 10, 7, 7, 7, 7, 0],
                 sample_rate=30000):
        super(AugmentData, self).__init__()
        self.batch_size = batch_size
        self.data_type = data_type
        self.label_index = label_index
        self.splits = splits
        self.sub_splits = sub_splits
        self.sample_rate = sample_rate
        if self.splits[0] + self.splits[1] != self.batch_size:
            print("Splits do not add up to one. Changed splits to 16 untoched and 48 augmented")
            self.splits = [16, 48]
        if sum(self.sub_splits) != self.splits[1]:
            print("Splits do not add up to one. Changed splits to [10, 10, 7, 7, 7, 7, 0] and the main split [16, 48].")
            self.splits = [16, 48]
            self.sub_splits = [10, 10, 7, 7, 7, 7, 0]
    
    def _split_data(self, data):  
        random.shuffle(data)
        return data[:self.splits[0]], data[self.splits[0]:]

    def _split_data_for_augment(self, data):
        splited_data = []
        for split_index, split in enumerate(self.sub_splits):
            if split_index == 0:
                splited_data.append(data[:self.sub_splits[0]])
                # print(self.sub_splits[0])
            elif split_index > 0 and split_index < len(self.sub_splits) - 1:
                splited_data.append(data[sum(self.sub_splits[:split_index]): sum(self.sub_splits[:split_index + 1])])
                # print(sum(self.sub_splits[:split_index]), sum(self.sub_splits[:split_index + 1]))
            else:
                splited_data.append(data[sum(self.sub_splits[:split_index]):])
                # print(sum(self.sub_splits[:split_index]))

        return splited_data

    def __random_temporal_shifting(self, data):
        timeshift_fac = 1 - 0.25 * (np.random.uniform()) # up to 25% of length
        shift_start = int(data[0].size()[1] * timeshift_fac)
        data[0] = torch.roll(data[0].float(), shifts=shift_start, dims=1)
        return data
    
    def __random_speed(self, data, min_speed=0.74, max_speed=1.35):
        tempo_factor = np.random.uniform(min_speed, max_speed)
        effects = [
            ["tempo", f"{tempo_factor}"],
        ]
        data[0], _ = torchaudio.sox_effects.apply_effects_tensor(data[0].float(), self.sample_rate, effects)
        return data

    def __random_pitch(self, data, min_shift=-100, max_shift=100):
        pitch_factor = np.int32(np.round(np.random.uniform(min_shift, max_shift)))
        effects = [
            ['pitch',f'{pitch_factor}'],
            ['rate', f'{self.sample_rate}']
        ]
        data[0], _ = torchaudio.sox_effects.apply_effects_tensor(data[0].float(), self.sample_rate, effects)
        return data
    
    def __random_gaussian_noise_uniform_ranged_magnetude(self, data, min_range=-0.01, max_range=0.01):
        magnetude = np.random.uniform(min_range, max_range)
        data[0] = data[0] + (magnetude * self.__normalize([torch.randn(data[0].size()), 0])[0]) 
        return data

    def __random_reverb(self, data):
        effects = [
            ['reverb', f'{np.int32(np.round(np.random.uniform(0, 100)))}',
            f'{np.int32(np.round(np.random.uniform(0, 100)))}',
            f'{np.int32(np.round(np.random.uniform(0, 100)))}',
            f'{np.int32(np.round(np.random.uniform(0, 100)))}',
            f'{np.int32(np.round(np.random.uniform(0, 100)))}',
            f'{np.int32(np.round(np.random.uniform(-10, 10)))}'],
            ['rate', f'{self.sample_rate}']
        ]
        data[0], _ = torchaudio.sox_effects.apply_effects_tensor(data[0].float(), self.sample_rate, effects)
        return data
    
    def __random_low_pass_filter_uniform_ranged_cutoff_frequency(self, data, min_frequency=50, max_frequency=600):
        cut_off_frequency = np.random.uniform(min_frequency, max_frequency)
        effects = [
            ['lowpass', '-2', f'{cut_off_frequency}'],
            ["rate", f"{self.sample_rate}"],
        ]
        data[0], _ = torchaudio.sox_effects.apply_effects_tensor(data[0].float(), self.sample_rate, effects)
        return data
    
    def __random_shutter_frequency_uniform_ranged_dutycycle(self, data, min_range=0.3, max_range=1.0):
        t = np.linspace(0, 2, data[0].size()[1], endpoint=False)
        duty_cycle = np.random.uniform(min_range, max_range)
        square_signal = (square(2 * np.pi * 30 * t, duty=duty_cycle) + 1) / 2
        data[0] = data[0].float() * torch.tensor(square_signal).float()
        concat_wave = torch.zeros(8, 60000, dtype=torch.float)
        for i in range(8):
            concat_channel_tensor = data[0][i][data[0][i] != 0]
            concat_wave[i][60000 - concat_channel_tensor.size()[0]:] = concat_channel_tensor
        effects = [
            ["rate", f"{1 / duty_cycle * self.sample_rate}"],
        ]
        data[0], _ = torchaudio.sox_effects.apply_effects_tensor(concat_wave, self.sample_rate, effects)
        return data

    def __swap_x_and_y(self, data):
        new_order = [4, 5, 6, 7] + [0, 1, 2, 3]
        data[0] = data[0][new_order]
        return data 

    def __swap_channels_in_x(self, data):
        new_order = random.sample([0, 1, 2, 3], 4) + [4, 5, 6, 7]
        data[0] = data[0][new_order]
        return data 

    def __swap_channels_in_y(self, data):
        new_order = [0, 1, 2, 3] + random.sample([4, 5, 6, 7], 4)
        data[0] = data[0][new_order]
        return data

    def __swap_channels_in_x_and_y(self, data):
        new_order = random.sample([0, 1, 2, 3], 4) + random.sample([4, 5, 6, 7], 4)
        data[0] = data[0][new_order]
        return data

    def __swap_x_and_y_and_channels_in_x(self, data):
        new_order = [4, 5, 6, 7] + random.sample([0, 1, 2, 3], 4) 
        data[0] = data[0][new_order]
        return data

    def __swap_x_and_y_and_channels_in_y(self, data):
        new_order = random.sample([4, 5, 6, 7], 4) + [0, 1, 2, 3] 
        data[0] = data[0][new_order]
        return data

    def __swap_x_and_y_and_both_channels(self, data):
        new_order = random.sample([4, 5, 6, 7], 4) + random.sample([0, 1, 2, 3], 4)
        data[0] = data[0][new_order]
        return data

    def __random_channel_swaping(self, data):
        swapping_mode = np.random.randint(0, 6)
        swap = {0: self.__swap_x_and_y, 1: self.__swap_channels_in_x, 
                2: self.__swap_channels_in_y, 3: self.__swap_channels_in_x_and_y,
                4: self.__swap_x_and_y_and_channels_in_x, 
                5: self.__swap_x_and_y_and_channels_in_y, 
                6: self.__swap_x_and_y_and_both_channels}
        return swap[swapping_mode](data)
    
    def __random_channel_swaping_with_gaussian_noise(self, data, min_range=-0.05, max_range=0.05):
        data = self.__random_channel_swaping(data)
        noise_magnetude = torch.tensor(np.random.uniform(min_range, max_range, (8, 1)))
        noise = noise_magnetude * torch.randn(data[0].size())
        data[0] = data[0] + noise
        return data

    def __random_mixed_channel_swaping_with_gaussian_noise(self, data, min_range=-0.5, max_range=0.5):
        new_order = random.sample(list(range(8)), 8)
        noise_magnetude = torch.tensor(np.random.uniform(min_range, max_range, (8, 1)))
        noise = noise_magnetude * torch.randn(data[0].size()) 
        data[0] = data[0][new_order] + noise
        return data
    
    def __random_base_uniform_range_gain_and_frequency(self, data, min_range=5, max_range=20, min_frequency=500, max_frequency=1000):
        base_gain = np.int32(np.round(np.random.uniform(min_range, max_range)))
        bass_frequency = np.int32(np.round(np.random.uniform(min_frequency, max_frequency)))
        effects = [
            ['bass', f'{base_gain}', f'{bass_frequency}'],  # bass shift
        ]
        data[0], _ = torchaudio.sox_effects.apply_effects_tensor(data[0].float(), self.sample_rate, effects)
        return data
    
    def __random_treble_uniform_range_gain_and_frequency(self, data, min_range=5, max_range=20, min_frequency=50, max_frequency=200):
        treble_gain = np.int32(np.round(np.random.uniform(min_range, max_range)))
        treble_frequency = np.int32(np.round(np.random.uniform(min_frequency, max_frequency)))
        effects = [
            ['treble', f'{treble_gain}', f'{treble_frequency}'],  # bass shift
        ]
        data[0], _ = torchaudio.sox_effects.apply_effects_tensor(data[0].float(), self.sample_rate, effects)
        return data
    
    def __normalize(self, data):
        effects = [
            ['gain', '-n'],  # normalises to 0dB
        ]
        data[0], _ = torchaudio.sox_effects.apply_effects_tensor(data[0].float(), self.sample_rate, effects)
        return data
    
    def __check_and_correct_length(self, data):
        if [data[0].size()[0], data[0].size()[1]] == [8, 60000]:
            return data
        effects = [
            ['reverse'],
            ['pad', '0', '1.0'], # add 1.0 seconds silence at the end
            ['trim', '0', '2'],  # get the first 2 seconds
            ['reverse'],
        ]
        data[0], _ = torchaudio.sox_effects.apply_effects_tensor(data[0].float(), self.sample_rate, effects)
        return data
    
    def __set_wave_type_to_float(self, data):
        data[0] = data[0].float()
        return data

    def __random_multiple_equalizer_uniform_range_gain_frequency_and_width(self, data, 
                                                                           filter_count_min=3, filter_count_max=10,
                                                                           min_gain=-20, max_gaine=20, 
                                                                           min_frequency=50, max_frequency=650,
                                                                           min_width=10, max_width=100):
        num_equalizers = np.int32(np.round(np.random.uniform(filter_count_min, filter_count_max)))
        waveform = copy.deepcopy(data[0].float())
        for i in range(num_equalizers):
            equalizers_gain = np.int32(np.round(np.random.uniform(min_gain, max_gaine)))
            equalizer_frequency = np.int32(np.round(np.random.uniform(min_frequency, max_frequency)))
            equalizer_width = np.int32(np.round(np.random.uniform(min_width, max_width)))
            effects = [
                ['equalizer', f'{equalizer_frequency}',
                f'{equalizer_width}', f'{equalizers_gain}'],  # bass shift
            ]
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(data[0].float(), self.sample_rate, effects)
        data[0] = waveform
        return data

    def _general_augmentation(self, data):
        method_index_for_each_sample = random.choices(list(range(5)), k=len(data))
        methods = [self.__random_temporal_shifting, 
                   self.__random_speed,
                   self.__random_pitch,
                   self.__random_gaussian_noise_uniform_ranged_magnetude,
                   self.__random_reverb]
        for i in range(len(data)):
            # print("HaHa general", len(data))
            data[i] = methods[method_index_for_each_sample[i]](data[i])
        return data

    def _volume_augmentation(self, data):
        method_index_for_each_sample = random.choices(list(range(2)), k=len(data))
        methods = [self.__random_gaussian_noise_uniform_ranged_magnetude,
                   self.__random_low_pass_filter_uniform_ranged_cutoff_frequency]
        for i in range(len(data)):
            # print("HaHa volume", len(data))
            if method_index_for_each_sample[i] == 0:
                data[i] = methods[method_index_for_each_sample[i]](data[i], min_range=-2.0, max_range=2.0)
            else: 
                data[i] = methods[method_index_for_each_sample[i]](data[i])
        return data
    
    def _device_augmentation(self, data):
        method_index_for_each_sample = random.choices(list(range(3)), k=len(data))
        methods = [self.__random_gaussian_noise_uniform_ranged_magnetude,
                   self.__random_low_pass_filter_uniform_ranged_cutoff_frequency,
                   self.__random_shutter_frequency_uniform_ranged_dutycycle]
        for i in range(len(data)):
            # print("HaHa device", len(data))
            if method_index_for_each_sample[i] == 0:
                data[i] = methods[method_index_for_each_sample[i]](data[i], min_range=-0.2, max_range=0.2)
            elif method_index_for_each_sample[i] == 1:
                data[i] = methods[method_index_for_each_sample[i]](data[i], min_frequency=20, max_frequency=1000)
            else:
                data[i] = methods[method_index_for_each_sample[i]](data[i])
        return data
    
    def _orientation_augmentation(self, data):
        method_index_for_each_sample = random.choices(list(range(2)), k=len(data))
        methods = [self.__random_channel_swaping, 
                   self.__random_channel_swaping_with_gaussian_noise]
        for i in range(len(data)):
            # print("HaHa orientation", len(data))
            data[i] = methods[method_index_for_each_sample[i]](data[i])
        return data
    
    def _background_augmentation(self, data):
        method_index_for_each_sample = random.choices(list(range(2)), k=len(data))
        methods = [self.__random_mixed_channel_swaping_with_gaussian_noise,
                   self.__random_low_pass_filter_uniform_ranged_cutoff_frequency]
        for i in range(len(data)):
            # print("HaHa background", len(data))
            if method_index_for_each_sample[i] == 0:
                data[i] = methods[method_index_for_each_sample[i]](data[i])
            else: 
                data[i] = methods[method_index_for_each_sample[i]](data[i], min_frequency=20, max_frequency=1000)
        return data
    
    def _surface_augmentation(self, data):
        method_index_for_each_sample = random.choices(list(range(2)), k=len(data))
        methods = [self.__random_channel_swaping_with_gaussian_noise, 
                   self.__random_low_pass_filter_uniform_ranged_cutoff_frequency]
        for i in range(len(data)):
            # print("HaHa surface", len(data))
            if method_index_for_each_sample[i] == 0:
                data[i] = methods[method_index_for_each_sample[i]](data[i], min_range=-0.5, max_range=0.5)
            else: 
                data[i] = methods[method_index_for_each_sample[i]](data[i], min_frequency=20, max_frequency=1000)
        return data
    
    def _digit_augmentation(self, data):
        method_index_for_each_sample = random.choices(list(range(4)), k=len(data))
        methods = [self.__random_pitch,
                   self.__random_base_uniform_range_gain_and_frequency,
                   self.__random_treble_uniform_range_gain_and_frequency,
                   self.__random_multiple_equalizer_uniform_range_gain_frequency_and_width]
        for i in range(len(data)):
            # print("HaHa digit", len(data))
            if method_index_for_each_sample[i] == 0:
                data[i] = methods[method_index_for_each_sample[i]](data[i], min_shift=-1000, max_shift=500)
            else: 
                data[i] = methods[method_index_for_each_sample[i]](data[i])
        return data
    
    def augment_data(self, data):
        if self.data_type != 'train':
            return data
        else:
            untuched_data, augmented_data = self._split_data(data)
            # print(f"Length of augmented data {len(augmented_data)} and untouched data {len(untuched_data)}")
            general_split, volume_split, device_split, orientation_split, background_split, surface_split, digit_split = self._split_data_for_augment(augmented_data)
            general_split = self._general_augmentation(general_split)
            volume_split = self._volume_augmentation(volume_split)
            device_split = self._device_augmentation(device_split)
            orientation_split = self._orientation_augmentation(orientation_split)
            background_split = self._background_augmentation(background_split)
            surface_split = self._surface_augmentation(surface_split)
            digit_split = self._digit_augmentation(digit_split)
            augmented_data = general_split + volume_split + device_split + orientation_split + background_split + surface_split + digit_split
            augmented_data = [self.__normalize(self.__check_and_correct_length(sample)) for sample in augmented_data]
        return [self.__set_wave_type_to_float(sample) for sample in random.sample(untuched_data + augmented_data, self.batch_size)]


            

class SpectralDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, batch_size, data_type='train', label_index=0):
        super(SpectralDataset, self).__init__()
        all_data_files = os.listdir(data_dir)
        all_data_files = sorted(all_data_files)
        cur_batch_sum = 0
        self.file_name_batch_Index = []
        self.data_files = []
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.label_index = label_index
        self.data_type = data_type
        for file_name in all_data_files:
            if data_type in file_name:
                print(f"Loading file {file_name} to size {data_type} set.")
                data_pkl = open(self.data_dir + '/' + file_name, 'rb')
                data_in_file = pickle.load(data_pkl)
                if len(data_in_file) // batch_size > 0:
                    cur_batch_sum += len(data_in_file) // batch_size
                    self.file_name_batch_Index.append((file_name, cur_batch_sum))
                    self.data_files.append(file_name)
                data_pkl.close()   
        print(f'{data_type.capitalize()} data includs the following files:' , self.file_name_batch_Index)

    def __len__(self):
        return self.file_name_batch_Index[-1][1]

    def load_file_data(self, index):
        data = []
        file_index = 0
        for i in range(len(self.file_name_batch_Index)):
            if self.file_name_batch_Index[i][1] >= index + 1:
                file_index = i
                break     
        data_pkl = open(self.data_dir + '/' + self.data_files[file_index], 'rb')
        cur_file_batch_num = index + 1 - self.file_name_batch_Index[file_index - 1][1] if file_index != 0 else index + 1
        data.extend(pickle.load(data_pkl)[((cur_file_batch_num - 1) * self.batch_size): (cur_file_batch_num  * self.batch_size)])
        '''if self.label_index != 2:
            augmentor = AugmentData(batch_size=64, data_type=self.data_type, label_index=self.label_index, splits=[16, 48], sub_splits=[48, 0, 0, 0, 0, 0, 0], sample_rate=30000)
        else:
            augmentor = AugmentData(batch_size=64, data_type=self.data_type, label_index=self.label_index, splits=[16, 48], sub_splits=[24, 0, 0, 0, 0, 0, 24], sample_rate=30000)

        data = augmentor.augment_data(data)'''
        file_data = [functional.lowpass_biquad(item[0].float(), 30000, 4000, 0.707) 
                     for item in data]

        file_label = [item[1][self.label_index].squeeze().long()
                      for item in data]
        data_pkl.close()
        return file_data, file_label 
        
    def __getitem__(self, index):
        return self.load_file_data(index)




class ResNetTypePretrainedModel(nn.Module):
    def __init__(
        self,
        torchvision_model,
        num_input_channels = 8,
        num_classes = 10,
        dropout = 0.02
    ):
        """
        EfficientNet main class

        Args:
            model_num (int): EfiicientNet model number that determins model structure  
            num_input_channels (int): Number of input channels
            num_classes (int): Number of classes
            dropout (float): dropout probability value        
        """
        super().__init__()
        self.best_accuracy = 0
        self.n_classes = num_classes
        self.pretrained_model = torchvision_model(pretrained=True)
        layer = self.pretrained_model.conv1
        new_layer = nn.Conv2d(in_channels=num_input_channels, 
                              out_channels=layer.out_channels, 
                              kernel_size=layer.kernel_size, 
                              stride=layer.stride, 
                              padding=layer.padding,
                              bias=layer.bias)

        copy_weights = 0 # Channel weights to copy from (RED in RGB)
        with torch.no_grad():
            new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

        for i in range(num_input_channels - layer.in_channels):
            channel = layer.in_channels + i
            with torch.no_grad():
                new_layer.weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights: copy_weights + 1, : :].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)

        self.pretrained_model.conv1 = new_layer
        for model_layer in [self.pretrained_model.layer1, 
                            self.pretrained_model.layer2,
                            self.pretrained_model.layer3, 
                            self.pretrained_model.layer4]:
            for block_index, block in enumerate(model_layer):
                if block_index == 0:
                    block.downsample[0].register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
                block.conv1.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
                block.conv2.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
                block.conv3.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))

        self.fc_classify = nn.Linear(1000, self.n_classes)

    def forward(self, x):
        return self.fc_classify(self.pretrained_model(x))

    def loss(self, prediction, label, reduction='mean'):
        if self.n_classes == 1:
            return nn.BCEWithLogitsLoss(reduction='mean')(prediction, label)
        else:
            return nn.CrossEntropyLoss()(prediction, label)

    # Saves the current model
    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    # Saves the best model so far
    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if accuracy > self.best_accuracy:
            self.save_model(file_path, num_to_keep)
            self.best_accuracy = accuracy

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)



class EfficientNetPretrainedModel(nn.Module):
    def __init__(
        self,
        torchvision_model,
        num_input_channels = 8,
        num_classes = 10,
        dropout = 0.02
    ):
        """
        EfficientNet main class

        Args:
            model_num (int): EfiicientNet model number that determins model structure  
            num_input_channels (int): Number of input channels
            num_classes (int): Number of classes
            dropout (float): dropout probability value        
        """
        super().__init__()
        self.best_accuracy = 0
        self.n_classes = num_classes
        self.pretrained_model = torchvision_model(pretrained=True)
        layer = self.pretrained_model.features[0][0]
        new_layer = nn.Conv2d(in_channels=num_input_channels, 
                              out_channels=layer.out_channels, 
                              kernel_size=layer.kernel_size, 
                              stride=layer.stride, 
                              padding=layer.padding,
                              bias=layer.bias)

        copy_weights = 0 # Channel weights to copy from (RED in RGB)
        with torch.no_grad():
            new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

        for i in range(num_input_channels - layer.in_channels):
            channel = layer.in_channels + i
            with torch.no_grad():
                new_layer.weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights: copy_weights + 1, : :].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)

        self.pretrained_model.features[0][0] = new_layer
        for feature_index, feature in enumerate(self.pretrained_model.features):
            if feature_index == 0:
                continue
            if feature_index + 1 == len(self.pretrained_model.features):
                continue
            for mbblock in feature:
                mbblock.block.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
        
        self.pretrained_model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
        self.fc_classify = nn.Linear(1000, self.n_classes)

    def forward(self, x):
        return self.fc_classify(self.pretrained_model(x))

    def loss(self, prediction, label, reduction='mean'):
        if self.n_classes == 1:
            return nn.BCEWithLogitsLoss(reduction='mean')(prediction, label)
        else:
            return nn.CrossEntropyLoss()(prediction, label)

    # Saves the current model
    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    # Saves the best model so far
    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if accuracy > self.best_accuracy:
            self.save_model(file_path, num_to_keep)
            self.best_accuracy = accuracy

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)


class PretrainedModelHuBERT(nn.Module):
    def __init__(
        self,
        model_name = "facebook/hubert-large-ls960-ft",
        num_input_channels = 8,
        num_classes = 10,
        dropout = 0.02
    ):
        """
        HuBERT class

        Args:
            model_num (int): EfiicientNet model number that determins model structure  
            num_input_channels (int): Number of input channels
            num_classes (int): Number of classes
            dropout (float): dropout probability value        
        """
        super().__init__()
        self.best_accuracy = 0
        self.n_classes = num_classes
        self.pretrained_model = HubertForSequenceClassification.from_pretrained(model_name)
        layer = self.pretrained_model.hubert.feature_extractor.conv_layers[0].conv
        new_layer = nn.Conv1d(in_channels=num_input_channels, 
                              out_channels=layer.out_channels, 
                              kernel_size=layer.kernel_size, 
                              stride=layer.stride, 
                              padding=layer.padding,
                              bias=True)

        copy_weights = 0 # Channel weights to copy from (RED in RGB)
        with torch.no_grad():
            new_layer.weight[:, :layer.in_channels, :] = layer.weight.clone()

        for i in range(num_input_channels - layer.in_channels):
            channel = layer.in_channels + i
            with torch.no_grad():
                new_layer.weight[:, channel:channel + 1, :] = layer.weight[:, copy_weights: copy_weights + 1, :].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)
        self.pretrained_model.hubert.feature_extractor.conv_layers[0].conv = new_layer
        self.pretrained_model.classifier = nn.Linear(256, self.n_classes)
        self.pretrained_model.hubert.feature_projection.dropout.p = dropout
        self.pretrained_model.hubert.encoder.dropout.p = dropout
        for layer_index, layer in enumerate(self.pretrained_model.hubert.encoder.layers):
            layer.dropout.p = dropout
            layer.feed_forward.intermediate_dropout.p = dropout
            layer.feed_forward.output_dropout.p = dropout

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        # HubertForSequenceClassification
        _HIDDEN_STATES_START_POSITION = 1
        return_dict = return_dict if return_dict is not None else self.pretrained_model.config.use_return_dict
        output_hidden_states = True if self.pretrained_model.config.use_weighted_layer_sum else output_hidden_states

        mask_time_indices=None
        output_attentions = output_attentions if output_attentions is not None else self.pretrained_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.pretrained_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.pretrained_model.config.use_return_dict
        # HubertFeatureExtractor
        hidden_states = input_values[:]
        if self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.pretrained_model.hubert.feature_extractor.conv_layers:
            if self.pretrained_model.hubert.feature_extractor._requires_grad and self.pretrained_model.hubert.feature_extractor.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(conv_layer),
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        extract_features = hidden_states
        # HubertMODEL
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self.pretrained_model.hubert._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.pretrained_model.hubert.feature_projection(extract_features)
        hidden_states = self.pretrained_model.hubert._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.pretrained_model.hubert.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        outputs = (hidden_states,) + encoder_outputs[1:]
        # HubertForSequenceClassification
        if self.pretrained_model.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.pretrained_model.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.pretrained_model.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self.pretrained_model._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.pretrained_model.classifier(pooled_output)

        loss = None
        output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
        return ((loss,) + output) if loss is not None else output[0]
    
    def loss(self, prediction, label, reduction='mean'):
        if self.n_classes == 1:
            return nn.BCEWithLogitsLoss(reduction='mean')(prediction, label)
        else:
            return nn.CrossEntropyLoss()(prediction, label)

    # Saves the current model
    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    # Saves the best model so far
    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if accuracy > self.best_accuracy:
            self.save_model(file_path, num_to_keep)
            self.best_accuracy = accuracy

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, epoch, PRINT_INTERVAL, data_loader, device, optimizer, pbar, pbar_update, BATCH_SIZE, logging_instant):
    model.to(device)
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data = torch.cat(data, dim=0)
        target = torch.cat(target, dim=0)
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = model.loss(output.squeeze(), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % PRINT_INTERVAL == 0:
            logging_instant.info(f"Train Epoch: {epoch} [{batch_idx * BATCH_SIZE}/{len(data_loader.dataset) * BATCH_SIZE} ({100. * (batch_idx * BATCH_SIZE) / (len(data_loader.dataset) * BATCH_SIZE):.0f}%)]\tLoss: {loss.item():.6f}")
        pbar.update(pbar_update)
        losses.append(loss.item())
    ave_losses = np.mean(losses)
    logging_instant.info(f"Train Epoch: {epoch} total average loss: {ave_losses:.6f}")
    return ave_losses


def train_binary(model, epoch, PRINT_INTERVAL, data_loader, device, optimizer, pbar, pbar_update, BATCH_SIZE, logging_instant):
    model.to(device)
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data = torch.cat(data, dim=0)
        target = torch.cat(target, dim=0)
        data = data.to(device)
        target = target.float()
        target = target.to(device)
        output = model(data)
        loss = model.loss(output.squeeze(), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % PRINT_INTERVAL == 0:
            logging_instant.info(f"Train Epoch: {epoch} [{batch_idx * BATCH_SIZE}/{len(data_loader.dataset) * BATCH_SIZE} ({100. * (batch_idx * BATCH_SIZE) / (len(data_loader.dataset) * BATCH_SIZE):.0f}%)]\tLoss: {loss.item():.6f}")
        pbar.update(pbar_update)
        losses.append(loss.item())
    ave_losses = np.mean(losses)
    logging_instant.info(f"Train Epoch: {epoch} total average loss: {ave_losses:.6f}")
    return ave_losses



def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

trained_and_tested_device = "trained on pixel 2 tested on Pixel 3"

def test(model, epoch, data_loader, device, BATCH_SIZE, logging_instant):
    model.to(device)
    model.eval()
    correct = 0
    losses = []
    all_pred = []
    all_proba = []
    all_target = []
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = torch.cat(data, dim=0)
            target = torch.cat(target, dim=0)
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            losses.append(model.loss(output.squeeze(), target).item())
            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
            all_proba.append(F.softmax(output, dim=1).cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_target.append(target.view_as(pred).cpu().numpy())
            if batch_idx % 20 == 0:
                logging_instant.info(f"Testing batch {batch_idx} of {len(data_loader.dataset)}")
    test_loss = np.mean(losses)
    conf_matrix = confusion_matrix(np.concatenate(all_target, axis=0), np.concatenate(all_pred, axis=0))
    logging_instant.info(f'Confusion matrix:\n{conf_matrix}')
    class_report = classification_report(np.concatenate(all_target, axis=0), np.concatenate(all_pred, axis=0))
    logging_instant.info(f'Classification Report:\n{class_report}')
    area_under_curv = roc_auc_score(np.concatenate(all_target, axis=0), np.concatenate(all_proba, axis=0).squeeze(), multi_class='ovr')
    logging_instant.info(f'Area Under The Curve:\n{area_under_curv}')
    logging_instant.info(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(data_loader.dataset) * BATCH_SIZE} ({100. * correct / (len(data_loader.dataset) * BATCH_SIZE):.2f}%). Average loss is: {test_loss:.6f}\n")
    test_accuracy = 100. * correct / len(data_loader.dataset)
    return test_loss, test_accuracy, area_under_curv


def test_binary(model, epoch, data_loader, device, BATCH_SIZE, logging_instant):
    model.to(device)
    model.eval()
    correct = 0
    losses = []
    all_pred = []
    all_proba = []
    all_target = []
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = torch.cat(data, dim=0)
            target = torch.cat(target, dim=0)
            target = target.float()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            losses.append(model.loss(output.squeeze(), target).item())
            pred = (output > 0.0).float()
            correct_mask = pred.eq(target.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            all_proba.append(torch.sigmoid(output).cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_target.append(target.view_as(pred).cpu().numpy())
            if batch_idx % 20 == 0:
                logging_instant.info(f"Testing batch {batch_idx} of {len(data_loader.dataset)}")
    test_loss = np.mean(losses)
    conf_matrix = confusion_matrix(np.concatenate(all_target, axis=0), np.concatenate(all_pred, axis=0))
    logging_instant.info(f'Confusion matrix:\n{conf_matrix}')
    class_report = classification_report(np.concatenate(all_target, axis=0), np.concatenate(all_pred, axis=0))
    logging_instant.info(f'Classification Report:\n{class_report}')
    area_under_curv = roc_auc_score(np.concatenate(all_target, axis=0), np.concatenate(all_proba, axis=0).squeeze(), multi_class='ovr')
    logging_instant.info(f'Area Under The Curve:\n{area_under_curv}')
    logging_instant.info(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(data_loader.dataset) * BATCH_SIZE} ({100. * correct / (len(data_loader.dataset) * BATCH_SIZE):.2f}%). Average loss is: {test_loss:.6f}\n")
    test_accuracy = 100. * correct / len(data_loader.dataset)
    return test_loss, test_accuracy, area_under_curv



def train_a_2D_CNN_spectral_classifier(BASE_PATH, DATA_PATH, logging_instant, device_instant, num_classes, label_index):    
    num_workers = 5
    EPOCHS = 5000
    BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    USE_CUDA = True
    LEARNING_RATE = 8e-5
    WEIGHT_DECAY = 0.00005
    PRINT_INTERVAL = 10
    SCHEDULER_EPOCH_STEP = 4
    SCHEDULER_GAMMA = 0.8

    LOG_PATH = BASE_PATH + '/logs/log' + '1' + '.pkl'

    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = device_instant
    logging_instant.info(f'Using device {device}')
    
    logging_instant.info(f'num workers: {num_workers}')

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}
    model = PretrainedModelHuBERT(num_input_channels=8, num_classes=num_classes, dropout=0.05, model_name="facebook/hubert-large-ls960-ft")
    model.to(device)
    logging_instant.info(f"{model}")
    logging_instant.info(f"Number of parameters: {count_parameters(model)}")
    train_set = SpectralDataset(DATA_PATH, BATCH_SIZE, data_type='train', label_index=label_index)
    test_set = SpectralDataset(DATA_PATH, VALID_BATCH_SIZE, data_type='test', label_index=label_index)
    valid_set = SpectralDataset(DATA_PATH, VALID_BATCH_SIZE, data_type='valid', label_index=label_index)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=True, drop_last=False, 
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=False, drop_last=False,
                                              **kwargs)

    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1,
                                              shuffle=False, drop_last=False,
                                              **kwargs)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                            weight_decay=WEIGHT_DECAY, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_EPOCH_STEP, gamma=SCHEDULER_GAMMA) # torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=9e-8, max_lr=1e-4, step_size_up=3, step_size_down=33, cycle_momentum=False, mode='exp_range', gamma=0.88) #
    start_epoch = model.load_last_model(BASE_PATH + '/checkpoints')

    train_losses, eval_losses, eval_accuracies, eval_area_under_curvs = pt_util.read_log(LOG_PATH, ([], [], [], []))
    eval_loss, eval_accuracy, eval_area_under_curv = 0, 0, 0
    if num_classes != 1:
        eval_loss, eval_accuracy, eval_area_under_curv = test(model, start_epoch, valid_loader, device, VALID_BATCH_SIZE, logging_instant)
    else:
        eval_loss, eval_accuracy, eval_area_under_curv = test_binary(model, start_epoch, valid_loader, device, VALID_BATCH_SIZE, logging_instant)
    model.best_accuracy = eval_area_under_curv
    eval_losses.append((start_epoch, eval_loss))
    eval_accuracies.append((start_epoch, eval_accuracy))
    eval_area_under_curvs.append((start_epoch, eval_area_under_curv))

    pbar_update = 1 / (len(train_loader) + len(test_loader))
    prev_learning_rate = LEARNING_RATE
    with tqdm(total=EPOCHS) as pbar:
        try:
             for epoch in range(start_epoch, EPOCHS + 1):
                train_loss, eval_loss, eval_accuracy, eval_area_under_curv = 0, 0, 0, 0
                if num_classes != 1:
                    train_loss = train(model, epoch, PRINT_INTERVAL, train_loader, device, optimizer, pbar, pbar_update, BATCH_SIZE, logging_instant)
                    eval_loss, eval_accuracy, eval_area_under_curv = test(model, epoch, valid_loader, device, VALID_BATCH_SIZE, logging_instant)
                else:
                    train_loss = train_binary(model, epoch, PRINT_INTERVAL, train_loader, device, optimizer, pbar, pbar_update, BATCH_SIZE, logging_instant)
                    eval_loss, eval_accuracy, eval_area_under_curv = test_binary(model, epoch, valid_loader, device, VALID_BATCH_SIZE, logging_instant)


                model.save_best_model(eval_area_under_curv, BASE_PATH + '/checkpoints/%03d.pt' % epoch, num_to_keep=5)
                scheduler.step()
                logging_instant.info(f"Current learning rate is: {optimizer.param_groups[0]['lr']}")
                '''if prev_learning_rate != optimizer.param_groups[0]['lr']:
                    start_epoch = model.load_last_model(BASE_PATH + '/checkpoints')
                    eval_loss, eval_accuracy = test(model, start_epoch, valid_loader, device)'''
                train_losses.append((epoch, train_loss))
                eval_losses.append((epoch, eval_loss))
                eval_accuracies.append((epoch, eval_accuracy))
                eval_area_under_curvs.append((epoch, eval_area_under_curv))
                pt_util.write_log(LOG_PATH, (train_losses, eval_losses, eval_accuracies, eval_area_under_curvs))
                prev_learning_rate = optimizer.param_groups[0]['lr']
                if epoch == 500:
                    break

        except KeyboardInterrupt as ke:
            logging_instant.info('Interrupted')
        except:
            import traceback
            traceback.print_exc()
        finally:
            logging_instant.info('Saving final model')
            model.save_model(BASE_PATH + '/checkpoints/f%03d.pt' % epoch, num_to_keep=5)
            x_val, y_val = zip(*train_losses)
            pt_util.plot(x_val, y_val, 'Train loss', 'Epoch', 'Training loss')
            x_val, y_val = zip(*eval_losses)
            pt_util.plot(x_val, y_val, 'Eval Losse', 'Epoch', 'Eval loss')
            x_val, y_val = zip(*eval_accuracies)
            pt_util.plot(x_val, y_val, 'Eval Accuracy', 'Epoch', 'Eval Accuracy')
            x_val, y_val = zip(*eval_area_under_curvs)
            pt_util.plot(x_val, y_val, 'Eval Area Under The Curv', 'Epoch', 'Area Under The Curv')
            return model, device



def test_a_2D_CNN_spectral_classifier(BASE_PATH, DATA_PATH, logging_instant, device_instant, num_classes, label_index):    
    num_workers = 5
    EPOCHS = 5000
    BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    USE_CUDA = True
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.00005
    PRINT_INTERVAL = 10
    SCHEDULER_EPOCH_STEP = 4
    SCHEDULER_GAMMA = 0.8

    LOG_PATH = BASE_PATH + '/logs/log' + '1' + '.pkl'

    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = device_instant
    logging_instant.info(f'Using device {device}')
    
    logging_instant.info(f'num workers: {num_workers}')

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}
    model = PretrainedModelHuBERT(num_input_channels=8, num_classes=num_classes, dropout=0.05, model_name="facebook/hubert-large-ls960-ft")

    model.to(device)
    logging_instant.info(f"{model}")
    logging_instant.info(f"Number of parameters: {count_parameters(model)}")
    test_set = SpectralDataset(DATA_PATH, VALID_BATCH_SIZE, data_type='test', label_index=label_index)
    valid_set = SpectralDataset(DATA_PATH, VALID_BATCH_SIZE, data_type='valid', label_index=label_index)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=False, drop_last=False,
                                              **kwargs)

    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1,
                                              shuffle=False, drop_last=False,
                                              **kwargs)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, 
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_EPOCH_STEP,
                                          gamma=SCHEDULER_GAMMA)
    start_epoch = model.load_last_model(BASE_PATH + '/checkpoints')

    train_losses, eval_losses, eval_accuracies, eval_area_under_curvs = pt_util.read_log(LOG_PATH, ([], [], [], []))

    logging_instant.info(f'Validation results of {trained_and_tested_device}:')
    eval_loss, eval_accuracy, eval_area_under_curv = 0, 0, 0
    if num_classes != 1:
        eval_loss, eval_accuracy, eval_area_under_curv = test(model, start_epoch, valid_loader, device, VALID_BATCH_SIZE, logging_instant)
    else:
        eval_loss, eval_accuracy, eval_area_under_curv = test_binary(model, start_epoch, valid_loader, device, VALID_BATCH_SIZE, logging_instant)

    logging_instant.info(f'Test results of {trained_and_tested_device}:')
    eval_loss, eval_accuracy, eval_area_under_curv = 0, 0, 0
    if num_classes != 1:
        eval_loss, eval_accuracy, eval_area_under_curv = test(model, start_epoch, test_loader, device, VALID_BATCH_SIZE, logging_instant)
    else:
        eval_loss, eval_accuracy, eval_area_under_curv = test_binary(model, start_epoch, test_loader, device, VALID_BATCH_SIZE, logging_instant)
    return model, device


def setup_logger(name_logfile, path_logfile):
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler = logging.FileHandler(path_logfile, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger


def spun_train_one_2D_CNN_spectral_classifier(device_instant, main_logger, type_label, num_classes, label_index):
    BASE_PATH = f'/blue/srampazzi/pnaghavi/test38_cross_device/classifier_{type_label}'
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        main_logger.info(f'Created BASE_PATH {BASE_PATH} for label instant {type_label}')
    DATA_PATH = '/blue/srampazzi/pnaghavi/test38_cross_device/EXT_DATA_AUDIO_IP'
    training_instant_logger = setup_logger(type_label + '_log.log', BASE_PATH + '/' + type_label + '_log.log')
    main_logger.info('Created logger instance with path ' + BASE_PATH + '/' + type_label + '_log.log' + 'for label instant ' + type_label)
    try:
        final_model, device = train_a_2D_CNN_spectral_classifier(BASE_PATH, DATA_PATH, training_instant_logger, device_instant, num_classes, label_index)
    except Exception as ex:
        main_logger.exception(ex)


def spun_test_one_2D_CNN_spectral_classifier(device_instant, main_logger, type_label, num_classes, label_index):
    BASE_PATH = f'/blue/srampazzi/pnaghavi/test38_cross_device/classifier_{type_label}'
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        main_logger.info(f'Created BASE_PATH {BASE_PATH} for label instant {type_label}')
    DATA_PATH = '/blue/srampazzi/pnaghavi/test38_cross_device/EXT_DATA_AUDIO_IP'
    training_instant_logger = setup_logger(type_label + '_log.log', BASE_PATH + '/' + type_label + '_log.log')
    main_logger.info('Created logger instance with path ' + BASE_PATH + '/' + type_label + '_log.log' + 'for label instant ' + type_label)
    try:
        final_model, device = test_a_2D_CNN_spectral_classifier(BASE_PATH, DATA_PATH, training_instant_logger, device_instant, num_classes, label_index)
    except Exception as ex:
        main_logger.exception(ex)


def main():
    typs_to_create_models_for = ['HuBERT_pretrained_filtered_DO05_ip12_digit', 'HuBERT_pretrained_filtered_DO05_ip12_gender', 'HuBERT_pretrained_filtered_DO05_ip12_speaker']
    class_counts = [10, 1, 20]
    main_logger = setup_logger(f'main_log.log', '/blue/srampazzi/pnaghavi/test38_cross_device/main_train_HuBERT_spectral_log_HUBERT_filtered_ip12_px5.log')
    typs_process_list = []
    if torch.cuda.device_count() != 3:
        main_logger.info(f"Not enough GPU was allocated 3 was requested but {torch.cuda.device_count()} was provided")
        return
    else:
        torch.multiprocessing.set_start_method('spawn')
        for index, type_str in enumerate(typs_to_create_models_for):
            typs_process_list.append(Process(target=spun_train_one_2D_CNN_spectral_classifier, args=('cuda:' + str(index), main_logger, type_str, class_counts[index], index)))
            # typs_process_list.append(Process(target=spun_test_one_2D_CNN_spectral_classifier, args=('cuda:' + str(index), main_logger, type_str, class_counts[index], index)))
            main_logger.info(f"Creating process for type {type_str}")
        for index, type_str in enumerate(typs_to_create_models_for):
            typs_process_list[index].start()
            main_logger.info(f"Starting process for type {type_str}")
            time.sleep(2)
        for index, type_str in enumerate(typs_to_create_models_for):
            typs_process_list[index].join()
            main_logger.info(f"Joining process pool for type {type_str}")
            time.sleep(2)


if __name__ == "__main__":
    main()
