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



class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "expand_ratio={expand_ratio}"
        s += ", kernel={kernel}"
        s += ", stride={stride}"
        s += ", input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_input_channels = 8,
        num_classes: int = 10,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()
        # _log_api_usage_once(self)
        self.best_accuracy = 0
        self.n_classes = num_classes

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                num_input_channels, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage.append(nn.Dropout(p=dropout, inplace=True))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout * 4, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

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


def _efficientnet(
    num_input_channels: int,
    num_classes: int,
    arch: str,
    width_mult: float,
    depth_mult: float,
    dropout: float,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> EfficientNet:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    model = EfficientNet(inverted_residual_setting, dropout, 
                         num_input_channels=num_input_channels, 
                         num_classes=num_classes, **kwargs)
    return model


def efficientnet_b0(num_input_channels: int=8, num_classes: int=10, dropout=0.2,
                    pretrained: bool = False, progress: bool = True,
                    **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_input_channels(int): Number of input channels of the data.
        num_classes(int): Number of classes of the data.
        dropout (float): The droupout probability
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(num_input_channels, num_classes,"efficientnet_b0",
                         1.0, 1.0, dropout, pretrained, progress, **kwargs)


def efficientnet_b1(num_input_channels: int=8, num_classes: int=10, dropout=0.2,
                    pretrained: bool = False, progress: bool = True,
                    **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B1 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_input_channels(int): Number of input channels of the data.
        num_classes(int): Number of classes of the data.
        dropout (float): The droupout probability
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(num_input_channels, num_classes,"efficientnet_b1",
                         1.0, 1.1, dropout, pretrained, progress, **kwargs)


def efficientnet_b2(num_input_channels: int=8, num_classes: int=10, dropout=0.3, 
                    pretrained: bool = False, progress: bool = True,
                    **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_input_channels(int): Number of input channels of the data.
        num_classes(int): Number of classes of the data.
        dropout (float): The droupout probability
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(num_input_channels, num_classes, "efficientnet_b2",
                         1.1, 1.2, dropout, pretrained, progress, **kwargs)



def efficientnet_b3(num_input_channels: int=8, num_classes: int=10, dropout=0.3,
                    pretrained: bool = False, progress: bool = True,
                    **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B3 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_input_channels(int): Number of input channels of the data.
        num_classes(int): Number of classes of the data.
        dropout (float): The droupout probability
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(num_input_channels, num_classes, "efficientnet_b3",
                         1.2, 1.4, dropout, pretrained, progress, **kwargs)



def efficientnet_b4(num_input_channels: int=8, num_classes: int=10, dropout=0.4, 
                     pretrained: bool = False, progress: bool = True,
                     **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_input_channels(int): Number of input channels of the data.
        num_classes(int): Number of classes of the data.
        dropout (float): The droupout probability
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(num_input_channels, num_classes, "efficientnet_b4",
                         1.4, 1.8, dropout, pretrained, progress, **kwargs)



def efficientnet_b5(num_input_channels: int=8, num_classes: int=10, dropout=0.4,
                    pretrained: bool = False, progress: bool = True, 
                    **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B5 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_input_channels(int): Number of input channels of the data.
        num_classes(int): Number of classes of the data.
        dropout (float): The droupout probability
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(num_input_channels, num_classes, "efficientnet_b5",
                         1.6, 2.2, dropout, pretrained, progress,
                         norm_layer=partial(nn.BatchNorm2d, eps=0.001,
                                            momentum=0.01), **kwargs)



def efficientnet_b6(num_input_channels: int=8, num_classes: int=10, dropout=0.5,
                     pretrained: bool = False, progress: bool = True, 
                     **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B6 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_input_channels(int): Number of input channels of the data.
        num_classes(int): Number of classes of the data.
        dropout (float): The droupout probability
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(num_input_channels, num_classes, "efficientnet_b6",
                         1.8, 2.6, dropout, pretrained, progress,
                         norm_layer=partial(nn.BatchNorm2d, eps=0.001,
                                            momentum=0.01), **kwargs)



def efficientnet_b7(num_input_channels: int=8, num_classes: int=10, dropout=0.5,
                    pretrained: bool = False, progress: bool = True,
                    **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B7 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_input_channels(int): Number of input channels of the data.
        num_classes(int): Number of classes of the data.
        dropout (float): The droupout probability
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(num_input_channels, num_classes, "efficientnet_b7",
                         2.0, 3.1, dropout, pretrained, progress,
                         norm_layer=partial(nn.BatchNorm2d, eps=0.001,
                                            momentum=0.01), **kwargs)

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
    num_workers = 20
    EPOCHS = 5000
    BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    USE_CUDA = True
    LEARNING_RATE = 5e-6
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
    model = efficientnet_b3(num_input_channels=8, num_classes=num_classes, dropout=0.01, pretrained=False, progress=True)
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
    model = efficientnet_b3(num_input_channels=8, num_classes=num_classes, dropout=0.025, pretrained=False, progress=True)
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
    typs_to_create_models_for = ['EfficientNet3_digit', 'EfficientNet3_gender', 'EfficientNet3_speaker']
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
