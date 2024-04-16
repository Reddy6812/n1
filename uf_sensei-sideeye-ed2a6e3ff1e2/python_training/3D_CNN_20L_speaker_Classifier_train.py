import os
import sys
import os
import pickle5 as pickle
import glob
import copy
import logging
import time

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
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchaudio

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import pt_util


class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, batch_size, data_type='train', label_index=0):
        super(BenchmarkDataset, self).__init__()
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
        file_data = [item[0].float()
                     for item in data]

        file_label = [item[1][self.label_index].squeeze().long() 
                      for item in data]
        data_pkl.close()
        return file_data, file_label 
        
    def __getitem__(self, index):
        return self.load_file_data(index)

class CNNDecoderClassifier(nn.Module):
    def __init__(self, n_input=3, n_output=1, stride=16, n_channel=8):
        super().__init__()
        self.best_accuracy = 0
        self.n_classes = n_output
        self.bn0 = nn.BatchNorm3d(n_input)

        self.conv1 = nn.Conv3d(n_input, n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn1 = nn.BatchNorm3d(n_channel)
        self.pool1 = nn.MaxPool3d(2)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv3d(n_channel, n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn2 = nn.BatchNorm3d(n_channel)
        self.pool2 = nn.MaxPool3d(2)
        self.dropout2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv3d(n_channel, 2 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn3 = nn.BatchNorm3d(2 * n_channel)
        self.pool3 = nn.MaxPool3d(2)
        self.dropout3 = nn.Dropout(0.1)

        self.conv4 = nn.Conv3d(2 * n_channel, 4 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn4 = nn.BatchNorm3d(4 * n_channel)
        self.pool4 = nn.MaxPool3d(2)
        self.dropout4 = nn.Dropout(0.1)

        self.conv5 = nn.Conv3d(4 * n_channel, 4 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn5 = nn.BatchNorm3d(4 * n_channel)
        self.pool5 = nn.MaxPool3d(2)
        self.dropout5 = nn.Dropout(0.1)

        self.conv6 = nn.Conv3d(4 * n_channel, 4 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn6 = nn.BatchNorm3d(4 * n_channel)
        self.pool6 = nn.MaxPool3d(2)
        self.dropout6 = nn.Dropout(0.1)

        self.conv7 = nn.Conv3d(4 * n_channel, 4 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn7 = nn.BatchNorm3d(4 * n_channel)
        self.pool7 = nn.MaxPool3d(2)
        self.dropout7 = nn.Dropout(0.1)

        self.conv8 = nn.Conv3d(4 * n_channel, 4 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn8 = nn.BatchNorm3d(4 * n_channel)
        self.pool8 = nn.MaxPool3d(2)
        self.dropout8 = nn.Dropout(0.1)

        self.conv9 = nn.Conv3d(4 * n_channel, 8 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn9 = nn.BatchNorm3d(8 * n_channel)
        self.pool9 = nn.MaxPool3d(2)
        self.dropout9 = nn.Dropout(0.1)

        self.conv10 = nn.Conv3d(8 * n_channel, 8 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn10 = nn.BatchNorm3d(8 * n_channel)
        self.pool10 = nn.MaxPool3d(2)
        self.dropout10 = nn.Dropout(0.1)

        self.conv11 = nn.Conv3d(8 * n_channel, 8 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn11 = nn.BatchNorm3d(8 * n_channel)
        self.pool11 = nn.AdaptiveMaxPool3d((4, 33, 4))
        self.dropout11 = nn.Dropout(0.1)

        self.conv12 = nn.Conv3d(8 * n_channel, 8 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn12 = nn.BatchNorm3d(8 * n_channel)
        self.pool12 = nn.MaxPool2d(2)
        self.dropout12 = nn.Dropout(0.1)

        self.conv13 = nn.Conv3d(8 * n_channel, 16 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn13 = nn.BatchNorm3d(16 * n_channel)
        self.pool13 = nn.AdaptiveMaxPool3d((4, 16, 4))
        self.dropout13 = nn.Dropout(0.1)
        
        self.conv14 = nn.Conv3d(16 * n_channel, 16 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn14 = nn.BatchNorm3d(16 * n_channel)
        self.pool14 = nn.MaxPool2d(2)
        self.dropout14 = nn.Dropout(0.1)

        self.conv15 = nn.Conv3d(16 * n_channel, 16 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn15 = nn.BatchNorm3d(16 * n_channel)
        self.pool15 = nn.AdaptiveMaxPool3d((4, 8, 4))
        self.dropout15 = nn.Dropout(0.1)

        self.conv16 = nn.Conv3d(16 * n_channel, 32 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn16 = nn.BatchNorm3d(32 * n_channel)
        self.pool16 = nn.MaxPool1d(2)
        self.dropout16 = nn.Dropout(0.1)
        
        self.conv17 = nn.Conv3d(32 * n_channel, 32 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn17 = nn.BatchNorm3d(32 * n_channel)
        self.pool17 = nn.AdaptiveMaxPool3d((4, 4, 4))
        self.dropout17 = nn.Dropout(0.1)

        self.conv18 = nn.Conv3d(32 * n_channel, 32 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn18 = nn.BatchNorm3d(32 * n_channel)
        self.pool18 = nn.MaxPool1d(2)
        self.dropout18 = nn.Dropout(0.1)

        self.conv19 = nn.Conv3d(32 * n_channel, 64 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn19 = nn.BatchNorm3d(64 * n_channel)
        self.pool19 = nn.MaxPool1d(2)
        self.dropout19 = nn.Dropout(0.1)

        self.conv20 = nn.Conv3d(64 * n_channel, 64 * n_channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn20 = nn.BatchNorm3d(64 * n_channel)
        self.pool20 = nn.MaxPool1d(2)

        self.avg = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(64 * n_channel, n_output)

    def forward(self, x):
        x = self.bn0(x)

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        # x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        # x = self.pool2(x)
        # x = self.dropout2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        # x = self.dropout3(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        # x = self.pool5(x)
        # x = self.dropout5(x)

        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        # x = self.pool6(x)

        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.pool7(x)

        x = self.conv8(x)
        x = F.relu(self.bn8(x))
        # x = self.pool8(x)
        # x = self.dropout8(x)
        
        x = self.conv9(x)
        x = F.relu(self.bn9(x))
        # x = self.pool9(x)
        # x = self.dropout9(x)

        x = self.conv10(x)
        x = F.relu(self.bn10(x))
        x = self.pool10(x)

        x = self.conv11(x)
        x = F.relu(self.bn11(x))
        x = self.pool11(x) 

        x = self.conv12(x)
        x = F.relu(self.bn12(x))
        # x = self.pool12(x)

        x = self.conv13(x)
        x = F.relu(self.bn13(x))
        x = self.pool13(x)
        
        x = self.conv14(x)
        x = F.relu(self.bn14(x))
        # x = self.pool14(x) 
        # x = self.dropout14(x)

        x = self.conv15(x)
        x = F.relu(self.bn15(x))
        x = self.pool15(x)
        # x = self.dropout15(x)

        x = self.conv16(x)
        x = F.relu(self.bn16(x))
        # x = self.pool16(x)

        x = self.conv17(x)
        x = F.relu(self.bn17(x))
        x = self.pool17(x)

        x = self.conv18(x)
        x = F.relu(self.bn18(x))
        # # x = self.pool18(x)
        # # x = self.dropout18(x)

        x = self.conv19(x)
        x = F.relu(self.bn19(x))
        # # x = self.pool19(x)
        # # x = self.dropout19(x)

        x = self.conv20(x)
        x = F.relu(self.bn20(x))
        # x = self.pool20(x)
        
        # x = torch.flatten(x, start_dim=1) 
        # print(x.size())
        x = torch.squeeze(self.avg(x))
        # print(x.size())
        # x = x.permute(0, 3, 2, 1)
        x = torch.unsqueeze(x, dim=1)
        # print(x.size())
        x = self.fc1(x)
        return x  

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
        data = torch.cat(data, dim=0).permute(0, 4, 2, 3, 1)
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
        data = torch.cat(data, dim=0).permute(0, 4, 2, 3, 1)
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
            data = torch.cat(data, dim=0).permute(0, 4, 2, 3, 1)
            target = torch.cat(target, dim=0)
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            losses.append(model.loss(output.squeeze(), target).item())
            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
            all_proba.append(F.softmax(output, dim=2).cpu().numpy())
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
            data = torch.cat(data, dim=0).permute(0, 4, 2, 3, 1)
            target = torch.cat(target, dim=0)
            target = target.float()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            losses.append(model.loss(output.squeeze(), target).item())
            pred = (output.squeeze() > 0.0).float()
            correct_mask = pred.eq(target.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            all_proba.append(torch.sigmoid(output.squeeze()).cpu().numpy())
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



def train_a_3D_CNN_decoder_classifier(BASE_PATH, DATA_PATH, logging_instant, device_instant, num_classes, label_index):    
    num_workers = 8
    EPOCHS = 5000
    BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    USE_CUDA = True
    LEARNING_RATE = 0.00002
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
    model = CNNDecoderClassifier(n_input=3, n_output=num_classes)
    model.to(device)
    logging_instant.info(f"{model}")
    logging_instant.info(f"Number of parameters: {count_parameters(model)}")
    train_set = BenchmarkDataset(DATA_PATH, VALID_BATCH_SIZE, data_type='train', label_index=label_index)
    test_set = BenchmarkDataset(DATA_PATH, VALID_BATCH_SIZE, data_type='test', label_index=label_index)
    valid_set = BenchmarkDataset(DATA_PATH, VALID_BATCH_SIZE, data_type='valid', label_index=label_index)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=True, drop_last=False, 
                                               **kwargs)

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



def test_a_3D_CNN_decoder_classifier(BASE_PATH, DATA_PATH, logging_instant, device_instant, num_classes, label_index):    
    num_workers = 8
    EPOCHS = 5000
    BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    USE_CUDA = True
    LEARNING_RATE = 0.0001
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
    model = CNNDecoderClassifier(n_input=3, n_output=num_classes)
    model.to(device)
    logging_instant.info(f"{model}")
    logging_instant.info(f"Number of parameters: {count_parameters(model)}")
    test_set = BenchmarkDataset(DATA_PATH, VALID_BATCH_SIZE, data_type='test', label_index=label_index)
    valid_set = BenchmarkDataset(DATA_PATH, VALID_BATCH_SIZE, data_type='valid', label_index=label_index)

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


def spun_train_one_3D_CNN_decoder_classifier(device_instant, main_logger, type_label, num_classes, label_index):
    BASE_PATH = f'/blue/srampazzi/pnaghavi/test30/3D_CNN_decoder_classifier_{type_label}'
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        main_logger.info(f'Created BASE_PATH {BASE_PATH} for label instant {type_label}')
    DATA_PATH = '/blue/srampazzi/pnaghavi/test30/EXT_DATA'
    training_instant_logger = setup_logger(type_label + '_log.log', BASE_PATH + '/' + type_label + '_log.log')
    main_logger.info('Created logger instance with path ' + BASE_PATH + '/' + type_label + '_log.log' + 'for label instant ' + type_label)
    try:
        final_model, device = train_a_3D_CNN_decoder_classifier(BASE_PATH, DATA_PATH, training_instant_logger, device_instant, num_classes, label_index)
    except Exception as ex:
        main_logger.exception(ex)


def spun_test_one_3D_CNN_decoder_classifier(device_instant, main_logger, type_label, num_classes, label_index):
    BASE_PATH = f'/blue/srampazzi/pnaghavi/test30/3D_CNN_decoder_classifier_{type_label}'
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        main_logger.info(f'Created BASE_PATH {BASE_PATH} for label instant {type_label}')
    DATA_PATH = '/blue/srampazzi/pnaghavi/test30/EXT_DATA'
    training_instant_logger = setup_logger(type_label + '_log.log', BASE_PATH + '/' + type_label + '_log.log')
    main_logger.info('Created logger instance with path ' + BASE_PATH + '/' + type_label + '_log.log' + 'for label instant ' + type_label)
    try:
        final_model, device = test_a_3D_CNN_decoder_classifier(BASE_PATH, DATA_PATH, training_instant_logger, device_instant, num_classes, label_index)
    except Exception as ex:
        main_logger.exception(ex)




def main():
    typs_to_create_models_for = ['20L_speaker_NODOUT']
    class_counts = [20]
    main_logger = setup_logger(f'main_log.log', '/blue/srampazzi/pnaghavi/test30/main_train_3D_CNN_log_2_s.log')
    typs_process_list = []
    if torch.cuda.device_count() != 1:
        main_logger.info(f"Not enough GPU was allocated 1 was requested but {torch.cuda.device_count()} was provided")
        return
    else:
        torch.multiprocessing.set_start_method('spawn')
        for index, type_str in enumerate(typs_to_create_models_for):
            typs_process_list.append(Process(target=spun_train_one_3D_CNN_decoder_classifier, args=('cuda:' + str(index), main_logger, type_str, class_counts[index], 2)))
            # typs_process_list.append(Process(target=spun_test_one_3D_CNN_decoder_classifier, args=('cuda:' + str(index), main_logger, type_str, class_counts[index], 2)))
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
