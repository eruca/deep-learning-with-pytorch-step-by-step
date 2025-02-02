import random
import datetime

import numpy as np
import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# 一个完全空的类


class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        # 这里定义类的属性

        # 首先将参数存储为属性，以供以后使用
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'mps'
        self.model.to(self.device)
        self.train_loader = None
        self.val_loader = None
        self.writer = None

        # 这些属性将在内部计算
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # hooks
        self.visualization = {}
        self.handlers = {}

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

    # def set_tensorboard(self, name, folder='runs'):
    #     # 此方法允许用户创建一个SummeryWriter以与TensorBoard交互
    #     suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    #     self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

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

    def _mini_batch(self, validation=False):
        # 小批量可以与两个加载器一起使用
        # 参数validation定义了将使用哪个加载器
        # 和相应地将要被使用的步骤函数
        if validation:
            data_loader = self.val_loader
            step = self.val_step
        else:
            data_loader = self.train_loader
            step = self.train_step

        if data_loader is None:
            return None

        # 设置好数据加载器和步骤函数
        # 这就是我们之前的小批量循环
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)

        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def train(self, n_epochs, seed=42):
        # 确保训练过程的可重复性
        self.set_seed(seed)

        for epoch in range(n_epochs):
            # 通过更新相应的属性来跟踪周期数
            self.total_epochs += 1

            # 内循环
            # 使用小批量执行训练
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            # 评估 -- 在评估期间不再需要梯度
            with torch.no_grad():
                # 使用小批量执行评估
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            # 如果设置了SummaryWriter....
            if self.writer is not None:
                scalars = {'training': loss}
                if self.val_losses is not None:
                    scalars.update({'validation': val_loss})

                self.writer.add_scalars(
                    main_tag="loss",
                    tag_scalar_dict=scalars,
                    global_step=epoch
                )
        if self.writer is not None:
            self.writer.flush()

    def save_checkpoint(self, filename):
        # 构建包含所有元素的字典以恢复训练
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses,
            'val_loss': self.val_losses
        }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # 加载字典
        checkpoint = torch.load(filename)

        # 恢复模型和优化器的状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train()  # always use TRAIN for resuming training

    def predict(self, x):
        # 设置为预测的评估模式
        self.model.eval()
        # 获取Numpy输入并使其成为一个浮点张量
        x_tensor = torch.as_tensor(x).float().to(self.device)
        # 将输入发送到设备并使用模型进行预测
        y_hat_tensor = self.model(x_tensor)
        # 将其设置会训练模式
        self.model.train()
        # 分离，将其带到CPU，并返回到Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Losses', c='b')
        if self.val_loader:
            plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def add_graph(self):
        if self.train_loader and self.writer:
            # 获取单个小批量，以便可以使用add_graph
            x_dummy, y_dummy = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_dummy.to(self.device))

    def count_parameters(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])

    @staticmethod
    def _visualize_tensors(axes, x, y=None, yhat=None, layer_name='', title=None, img_value=True):
        # 图像的数量是每行中'子图'的数量
        n_images = len(axes)
        # 获取缩放灰度的最大值和最小值
        minv, maxv = np.min(x), np.max(x)
        # 为每幅图
        for j, image in enumerate(x[:n_images]):
            ax = axes[j]
            # 设置标题、标签，并删除刻度
            if title is not None:
                ax.set_title(f'{title} #{j}', fontsize=12)
            shp = np.atleast_2d(image).shape
            ax.set_ylabel(
                f'{layer_name}\n{shp[0]}x{shp[1]}', rotation=0, labelpad=40)

            xlabel1 = '' if y is None else f"\nLabel: {y[j]}"
            xlabel2 = '' if yhat is None else f"\nPrediction: {yhat[j]}"
            xlabel = f'{xlabel1}{xlabel2}'
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # 将权重绘制为图像
            ax.imshow(np.atleast_2d(image.squeeze()),
                      cmap='gray', vmin=minv, vmax=maxv)

            if img_value:
                # 在每个方格中显示数值
                image_2d = np.atleast_2d(image.squeeze())
                for (ii, jj), val in np.ndenumerate(image_2d):
                    # ax.text 的坐标是 (x, y)，其中 x 是列索引，y 是行索引
                    ax.text(jj, ii, f'{val:.2f}', ha='center',
                            va='center', color='red', fontsize=8)

    def visualize_filter(self, layer_name, **kwargs):
        """layer_name: layer的名称, kwargs: {"img_value": bool=True}"""
        try:
            # 从模型中获取层对象
            layer = self.model
            for name in layer_name.split('.'):
                layer = getattr(layer, name)
            # 只关注2D卷积的滤波器
            if isinstance(layer, nn.Conv2d):
                # 获取权重
                weights = layer.weight.data.cpu().numpy()
                # 权重 -> (输出通道(滤波器), 输入通道, H, W)
                n_filters, n_channels, _, _ = weights.shape

                # 构建图像
                size = (2 * n_channels + 2, 2 * n_filters)
                fig, axes = plt.subplots(n_filters, n_channels, figsize=size)
                axes = np.atleast_2d(axes)

                # 遍历每个通道（滤波器）
                for i in range(n_filters):
                    StepByStep._visualize_tensors(
                        axes[i, :],
                        weights[i],
                        layer_name=f'Filter #{i}',
                        title='Channel',
                        img_value=kwargs['img_value']
                    )

                fig.tight_layout()
                for ax in axes.flat:
                    ax.label_outer()

                return fig
        except AttributeError:
            return

    def attach_hooks(self, layers_to_hook, hook_fn=None):
        # 清除任何以前的值
        self.visualization = {}
        # 创建字典以将层对象映射到他们的名称
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules}

        if hook_fn is None:
            def hook_fn(layer, inputs, outputs):
                # 获取层名称
                name = layer_names[layer]
                # 分离输出
                values = outputs.detach().cpu().numpy()
                # 由于钩子函数可能会被多次调用
                # 例如，如果对多个小批量进行预测
                # 处理连接结果
                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([
                        self.visualization[name], values
                    ])

        for name, layer in modules:
            # 如果层在列表中
            if name in layers_to_hook:
                # 初始化字典中对应的键
                self.visualization[name] = None
                # 注册前向钩子并将句柄保留在另外一个字典中
                self.handlers[name] = layer.register_forward_hook(hook_fn)

    def remove_hooks(self):
        # 删除所有钩子
        for name, handler in self.handlers.items():
            handler.remove()
        # 清除字典
        self.handlers = {}

    def visualize_outputs(self, layers, n_images=10, y=None, y_hat=None):
        layers = [l for l in layers if l in self.visualization.keys()]
        shapes = [self.visualization[layer].shape for layer in layers]

        n_rows = []
        for shape in shapes:
            if len(shape) == 4:  # 4 维输出 (batch_size, channels, height, width)
                n_rows.append(shape[1])  # 通道数
            elif len(shape) == 2:  # 2 维输出 (batch_size, features)
                n_rows.append(1)
            else:
                raise ValueError(f"Unsupported shape: {shape}")

        total_rows = np.sum(n_rows)

        fig, axes = plt.subplots(total_rows, n_images, figsize=(
            1.5 * n_images, 1.5 * total_rows))
        # axes = np.atleast_2d(axes)

        # 层层循环，每行子图一层
        current_row = 0
        for i, layer in enumerate(layers):
            # 为该层获取生成的特征图
            output = self.visualization[layer]
            is_vector = len(output.shape) == 2

            for j in range(n_rows[i]):
                StepByStep._visualize_tensors(
                    axes[current_row, :],
                    output if is_vector else output[:, j].squeeze(),
                    y,
                    y_hat,
                    layer_name=layers[i],
                    title='Image' if current_row == 0 else None,
                    img_value=False
                )
                current_row += 1

        for ax in axes.flat:
            ax.label_outer()
        fig.tight_layout()

        return fig

    def correct(self, x, y, threshold=0.5):
        # 将输入数据和标签移动到指定设备
        x, y = x.to(self.device), y.to(self.device)

        # 在推理模式下进行预测
        with torch.inference_mode():
            yhat = self.model(x)

        # 得到批量的大小和类的数量
        # (只有1， 如果它是二元的)
        n_samples, n_dims = yhat.shape

        if n_dims > 1:
            # 在多分类中，最大的logit总是获胜
            # 所以不必费心去获取概率

            # 这是Pytorch的argmax版本
            # 但它返回一个元组：（最大值，最大值的索引）
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims = 2  # 二元分类

            # 在二元分类中，需要检查最后一层是否是sigmoid(然后它会产生概率)
            # if isinstance(self.model, nn.Sequential) and isinstance(self.model[-1], nn.Sigmoid):
            if yhat.min() >= 0 and yhat.max() <= 1:  # yhat 已经是概率
                predicted = (yhat > threshold).long()
            else:
                predicted = (F.sigmoid(yhat) > threshold).long()

        # 每个类别正确分类了多少个样本
        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y == c] == c).sum().item()
            result.append((n_correct, n_class))

        return torch.tensor(result)

    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        results = torch.stack(results, axis=0)

        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.float().mean(axis=0)

        return results

    @staticmethod
    def statistics_per_channel(images, labels):
        # 获取输入图像的形状 (n_samples, n_channels, n_height, n_width)
        n_samples, n_channels, n_height, n_width = images.size()

        # 将每个通道的像素展平为 1 维
        flatten_per_channel = images.reshape(n_samples, n_channels, -1)

        # 计算每个通道每幅图像的统计数据
        # 每个通道像素的平均值 (n_samples, n_channels)
        means = flatten_per_channel.mean(axis=2)
        # 每个通道像素的标准差 (n_samples, n_channels)
        stds = flatten_per_channel.std(axis=2)

        # 计算整个小批量中所有图像的统计量
        sum_means = means.sum(axis=0)  # 所有图像在每个通道上的像素平均值之和 (n_channels,)
        sum_stds = stds.sum(axis=0)    # 所有图像在每个通道上的像素标准差之和 (n_channels,)
        n_samples = torch.tensor(
            [n_samples] * n_channels).float()  # 样本数量 (n_channels,)

        # 将统计量堆叠在一起 (3, n_channels)
        return torch.stack([n_samples, sum_means, sum_stds], axis=0)

    @staticmethod
    def make_normalizer(loader):
        total_samples, total_means, total_stds = \
            StepByStep.loader_apply(loader, StepByStep.statistics_per_channel)
        norm_mean = total_means / total_samples
        norm_std = total_stds / total_samples

        return Normalize(mean=norm_mean, std=norm_std)
