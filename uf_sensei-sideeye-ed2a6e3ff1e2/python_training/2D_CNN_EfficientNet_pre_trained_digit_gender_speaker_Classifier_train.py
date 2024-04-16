import os
import sys
import os
import math
import pickle5 as pickle
import glob
import copy
import logging
import time
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

import pt_util


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
        file_data = [torch.tensor(np.concatenate([(transforms.Spectrogram(n_fft=4000, win_length=4000, 
                                                            hop_length=2000)(item[0].float())).numpy()[:, :140, :],
                                                           (transforms.Spectrogram(n_fft=2000, win_length=2000, 
                                                            hop_length=1000)(item[0].float())).numpy()[:, :140, :],
                                                           (transforms.Spectrogram(n_fft=1000, win_length=1000, 
                                                            hop_length=500)(item[0].float())).numpy()[:, :140, :]], axis=2)) 
                     for item in data]
        file_label = [item[1][self.label_index].squeeze().long()
                      for item in data]
        data_pkl.close()
        return file_data, file_label 
        
    def __getitem__(self, index):
        return self.load_file_data(index)




class EfficientNetModel(nn.Module):
    def __init__(
        self,
        model_num: int = 4,
        num_input_channels: int = 8,
        num_classes: int = 10,
        dropout: float = 0.02
    ) -> None:
        """
        EfficientNet main class

        Args:
            model_num (int): EfiicientNet model number that determins model structure  
            num_input_channels (int): Number of input channels
            num_classes (int): Number of classes
            dropout (float): dropout probability value        
        """
        super().__init__()
        # _log_api_usage_once(self)
        self.best_accuracy = 0
        self.n_classes = num_classes
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b' + str(model_num), in_channels=num_input_channels, num_classes=num_classes)
        for block in self.efficientnet._blocks:
            block.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))

    def forward(self, x: Tensor) -> Tensor:
        return self.efficientnet(x)

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
    num_workers = 16
    EPOCHS = 5000
    BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    USE_CUDA = True
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.00005
    PRINT_INTERVAL = 10
    SCHEDULER_EPOCH_STEP = 10
    SCHEDULER_GAMMA = 0.8

    LOG_PATH = BASE_PATH + '/logs/log' + '1' + '.pkl'

    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = device_instant
    logging_instant.info(f'Using device {device}')
    
    logging_instant.info(f'num workers: {num_workers}')

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}
    model = EfficientNetModel(model_num=4, num_input_channels=8, num_classes=num_classes, dropout=0.005)
    model.to(device)
    logging_instant.info(f"{model}")
    logging_instant.info(f"Number of parameters: {count_parameters(model)}")
    train_set = SpectralDataset(DATA_PATH, VALID_BATCH_SIZE, data_type='train', label_index=label_index)
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_EPOCH_STEP,
                                          gamma=SCHEDULER_GAMMA)
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
    num_workers = 20
    EPOCHS = 5000
    BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    USE_CUDA = True
    LEARNING_RATE = 2e-5
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
    model = EfficientNetModel(model_num=4, num_input_channels=8, num_classes=num_classes, dropout=0.005)   
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

    logging_instant.info(f'Validation results:')
    eval_loss, eval_accuracy, eval_area_under_curv = 0, 0, 0
    if num_classes != 1:
        eval_loss, eval_accuracy, eval_area_under_curv = test(model, start_epoch, valid_loader, device, VALID_BATCH_SIZE, logging_instant)
    else:
        eval_loss, eval_accuracy, eval_area_under_curv = test_binary(model, start_epoch, valid_loader, device, VALID_BATCH_SIZE, logging_instant)

    logging_instant.info(f'Test results:')
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
    BASE_PATH = f'/blue/srampazzi/pnaghavi/test30/2D_CNN_classifier_{type_label}'
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        main_logger.info(f'Created BASE_PATH {BASE_PATH} for label instant {type_label}')
    DATA_PATH = '/blue/srampazzi/pnaghavi/test30/EXT_DATA_AUDIO'
    training_instant_logger = setup_logger(type_label + '_log.log', BASE_PATH + '/' + type_label + '_log.log')
    main_logger.info('Created logger instance with path ' + BASE_PATH + '/' + type_label + '_log.log' + 'for label instant ' + type_label)
    try:
        final_model, device = train_a_2D_CNN_spectral_classifier(BASE_PATH, DATA_PATH, training_instant_logger, device_instant, num_classes, label_index)
    except Exception as ex:
        main_logger.exception(ex)


def spun_test_one_2D_CNN_spectral_classifier(device_instant, main_logger, type_label, num_classes, label_index):
    BASE_PATH = f'/blue/srampazzi/pnaghavi/test30/2D_CNN_classifier_{type_label}'
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        main_logger.info(f'Created BASE_PATH {BASE_PATH} for label instant {type_label}')
    DATA_PATH = '/blue/srampazzi/pnaghavi/test30/EXT_DATA_AUDIO'
    training_instant_logger = setup_logger(type_label + '_log.log', BASE_PATH + '/' + type_label + '_log.log')
    main_logger.info('Created logger instance with path ' + BASE_PATH + '/' + type_label + '_log.log' + 'for label instant ' + type_label)
    try:
        final_model, device = test_a_2D_CNN_spectral_classifier(BASE_PATH, DATA_PATH, training_instant_logger, device_instant, num_classes, label_index)
    except Exception as ex:
        main_logger.exception(ex)




def main():
    typs_to_create_models_for = ['EfficientNet4_pretrained_dropout_digit', 'EfficientNet4_dropout_pretrained_gender', 'EfficientNet4_dropout_pretrained_speaker']
    class_counts = [10, 1, 20]
    main_logger = setup_logger(f'main_log.log', '/blue/srampazzi/pnaghavi/test30/main_train_2D_CNN_spectral_log_effNet.log')
    typs_process_list = []
    if torch.cuda.device_count() != 3:
        main_logger.info(f"Not enough GPU was allocated 3 was requested but {torch.cuda.device_count()} was provided")
        return
    else:
        torch.multiprocessing.set_start_method('spawn')
        for index, type_str in enumerate(typs_to_create_models_for):
            # typs_process_list.append(Process(target=spun_train_one_2D_CNN_spectral_classifier, args=('cuda:' + str(index), main_logger, type_str, class_counts[index], index)))
            typs_process_list.append(Process(target=spun_test_one_2D_CNN_spectral_classifier, args=('cuda:' + str(index), main_logger, type_str, class_counts[index], index)))
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
