import numpy as np
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard.writer import SummaryWriter

import matplotlib.pyplot as plt

# 一个完全空的类


class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        # 这里定义类的属性

        # 首先将参数存储为属性，以供以后使用
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.train_loader = None
        self.val_loader = None
        self.writer = None

        # 这些属性将在内部计算
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # 为模型、损失函数和优化器创建train_step函数
        # 注意：哪里没有参数，它直接使用类属性
        self.train_step = self._make_train_step()
        # 为模型和损失函数创建val_step函数
        self.val_step = self._make_val_step()

    def to(self, device):
        # 此方法允许用户指定不同的设备
        # 它设置相应的属性（稍后在小批量中使用)并将模型发送到设备
        self.device = device
        self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        # 此方法允许用户定义使用哪个train_loader(和val_loader,可选)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder='runs'):
        # 此方法允许用户创建一个SummeryWriter以与TensorBoard交互
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step(self):
        # 这个方法不需要ARGS
        # 可以参考属性:self.model, self.loss_fn和self.optimizer

        # 构建在训练循环中执行一个步骤的函数
        def perform_train_step(x, y):
            # 设置模型为训练模式
            self.model.train()

            # 第1步：计算模型的预测输出--前向传递
            yhat = self.model(x)
            # 第2步：计算损失
            loss = self.loss_fn(yhat, y)
            # 第3步：计算参数b和w的梯度
            loss.backward()

            # 第4步：使用梯度和学习率更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return perform_train_step

    def _make_val_step(self):
        # 构建在验证循环中执行步骤的函数
        def perform_val_step(x, y):
            # 设置模型为评估模式
            self.model.eval()

            # 第1步：计算模型的预测输出--前向传递
            yhat = self.model(x)
            # 第2步：计算损失
            loss = self.loss_fn(yhat, y)

            return loss.item()

        return perform_val_step
