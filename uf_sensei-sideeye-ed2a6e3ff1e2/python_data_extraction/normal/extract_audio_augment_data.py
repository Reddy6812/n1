from multiprocessing import Pool, TimeoutError
import glob
import os
import copy
import random
import pickle5 as pickle

from scipy.io.wavfile import read

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import multiplenoisereduce


dataset_device_scheme = 's20+_PF_'
dir_path = '/orange/vbindschaedler/pnaghavi/Data/test38_s20+/audio_PF/'
drop_path = '/blue/srampazzi/pnaghavi/test38_cross_device/EXT_DATA_AUDIO_s20+/'
CSV_path = '/orange/vbindschaedler/pnaghavi/Data/test38_s20+/wordlog_py.csv'
BATCH_SIZE = 64
Num_worker = 50
MAX_LENGTH = 60000
SAMPLE_RATE = 30000


files = glob.glob(os.path.join(dir_path, '*.wav'))
file_names = copy.deepcopy(files)
file_names = [file_name.split(dir_path)[1].split('.')[0] for file_name in file_names]

label_dict = {}
file_to_label_dict = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5,
                      '07': 6, '08': 7, '09': 8, '10': 9,'12': 10, '26': 11, 
                      '28': 12, '36': 13, '43': 14, '47': 15,
                      '52': 16, '56': 17, '57': 18, '58': 19}
# label_dict = {item: (int(item.split('-')[3]), 1) for item in file_names}
splits = []
with open(CSV_path, 'r') as f:
    for line in f:
        line_item = line.split(',')
        line_items_strip = [item.strip() for item in line_item]
        label_dict[line_items_strip[1].split('.')[0]] =\
        (int(line_items_strip[6]), 1 if line_items_strip[5] == 'female' else 0, file_to_label_dict[line_items_strip[1].split('.')[0].split('-')[2]])
        splits.append([line_items_strip[1].split('.')[0], int(line_items_strip[10])])

train_split, valid_split, test_split = [], [], []
for wave_file in splits:
    if wave_file[1] == 1:
        train_split.append(wave_file)
    elif wave_file[1] == 2:
        valid_split.append(wave_file)
    else:
        test_split.append(wave_file)
ordered_split = valid_split + test_split + train_split
print(ordered_split)
ordered_split_dict = {file_split_pair[0]: file_split_pair[1] for file_split_pair in ordered_split}

files_unordered = glob.glob(os.path.join(dir_path, '*.wav'))
file_names_unordered = copy.deepcopy(files_unordered)
file_names_unordered = {file_name.split(dir_path)[1].split('.')[0]: file_index for file_index, file_name in enumerate(file_names_unordered)}
files, file_names = [], []
for wave_file in ordered_split:
    files.append(files_unordered[file_names_unordered[wave_file[0]]])
    file_names.append(wave_file[0])
print(file_names)

count = 0
for file_name in file_names:
    if file_name.split('.')[0] not in label_dict:
        count += 1
        print(file_name.split('.')[0])

print(label_dict)

print(len(label_dict))

dataset = []
sample_rate = -1
max_squence_length = 0
def extract_batch(i):
    global BATCH_SIZE, files, file_names, MAX_LENGTH, drop_path, sample_rate, max_squence_length
    dataset = []
    print(len(files[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]))
    for f in range(len(files[i * BATCH_SIZE: (i + 1) * BATCH_SIZE])):
        if files[i * BATCH_SIZE + f].split(dir_path)[1].split('.')[0] != file_names[i * BATCH_SIZE + f]:
            print('Error: the order is off.')
            break
        data = read(files[i * BATCH_SIZE + f])
        data = list(data)
        if sample_rate == -1:
            sample_rate = data[0]
        else:
            if sample_rate != data[0]:
                print('Error: Sample rates don\'t match.')
                break
        np_wave_reduced_noise = multiplenoisereduce.subRSNoise(data[1], SAMPLE_RATE).swapaxes(1, 0)
        for channel_index in range(8):
            if max(np.abs(np_wave_reduced_noise[channel_index, :])) > 0:
                np_wave_reduced_noise[channel_index, :] /= max(np.abs(np_wave_reduced_noise[channel_index, :]))
            else:
                print(f"file: {file_names[i * BATCH_SIZE + f]} channel {channel_index} absolute max is zero")
        data = (sample_rate, np_wave_reduced_noise.swapaxes(1, 0))
        if data[1].shape[0] > MAX_LENGTH:
            print(f"File number {i * BATCH_SIZE + f + 1} is largeer than max length so it was"
                  f" shortened. Initial length was {data[1].shape[0]}.")
            data = (sample_rate, data[1][(data[1].shape[0] - MAX_LENGTH):, :])
        
        start_index = MAX_LENGTH - data[1].shape[0]
        tens_wave = torch.zeros(8, MAX_LENGTH, dtype=float)
        for j in range(data[1].shape[0]):
            for k in range(data[1].shape[1]):
                if j >= MAX_LENGTH:
                    print('Error: MAX_LENGTH is too small.')
                    break
                tens_wave[k][j + start_index] = data[1][j][k]
        max_squence_length = tens_wave.size()[1]\
                            if max_squence_length < tens_wave.size()[1]\
                            else max_squence_length
        dataset.append([tens_wave, (torch.tensor([label_dict[file_names[ i * BATCH_SIZE + f]][0]]),
                                    torch.tensor([label_dict[file_names[ i * BATCH_SIZE + f]][1]]),
                                    torch.tensor([label_dict[file_names[ i * BATCH_SIZE + f]][2]]),
                                    file_names[i * BATCH_SIZE + f])])
        print(i * BATCH_SIZE + f, i, f, file_names[i * BATCH_SIZE + f], data[1].shape)
        if (i * BATCH_SIZE + f) % 50 == 0:
            print(f"The current file being processed is:"
                  f" {(i * BATCH_SIZE + f) + 1} from total of {len(files)}")
    random.shuffle(dataset)
    random.shuffle(dataset)
    random.shuffle(dataset)
    random.shuffle(dataset)
    random.shuffle(dataset)
    batch_split = ordered_split_dict[dataset[0][1][3]]
    for data_row in dataset:
        if ordered_split_dict[data_row[1][3]] != batch_split:
            print("Error: Batch order is off. Splits were mixed.")
            return
    if batch_split == 1:
        pickle.dump(dataset, open(drop_path + dataset_device_scheme + f'data_{i + 1}_train.pkl', 'wb'))
    elif batch_split == 2:
        pickle.dump(dataset, open(drop_path + dataset_device_scheme + f'data_{i + 1}_valid.pkl', 'wb'))
    else:
        pickle.dump(dataset, open(drop_path + dataset_device_scheme +  f'data_{i + 1}_test.pkl', 'wb'))


with Pool(processes=Num_worker) as pool:
    for i in pool.imap_unordered(extract_batch, range(len(files) // BATCH_SIZE)):
        pass    

pkl_files = glob.glob(os.path.join(drop_path, '*.pkl'))
count_train = 0
count_test = 0
count_valid = 0
for item in pkl_files:
    if '_train.pkl' in item:
         count_train += 1
    elif '_valid.pkl' in item:
         count_valid += 1
    elif '_test.pkl' in item:
         count_test += 1
print(f'Training includes {count_train} batches and {count_train * BATCH_SIZE} rows!')
print(f'Validation includes {count_valid} batches and {count_valid * BATCH_SIZE} rows!')
print(f'test includes {count_test} batches and {count_test * BATCH_SIZE} rows!')