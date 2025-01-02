import random
import datetime

import numpy as np
import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


# 针对pytorch 2.0版本的代码
class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.model.to(self.device)
        self.train_loader = None
        self.val_loader = None
        self.writer = None

        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        self.visualization = {}
        self.handlers = {}

        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    # def set_tensorboard(self, name, folder='runs'):
    #     suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    #     self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step(self):
        def perform_train_step(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()

        return perform_train_step

    def _make_val_step(self):
        def perform_val_step(x, y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            return loss.item()

        return perform_val_step

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step = self.val_step
        else:
            data_loader = self.train_loader
            step = self.train_step

        if data_loader is None:
            return None

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
        self.set_seed(seed)

        for epoch in range(n_epochs):
            self.total_epochs += 1

            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

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
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses,
            'val_loss': self.val_losses
        }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train()

    def predict(self, x):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float().to(self.device)
        y_hat_tensor = self.model(x_tensor)
        self.model.train()
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
            x_dummy, y_dummy = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_dummy.to(self.device))

    def count_parameters(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])

    @staticmethod
    def _visualize_tensors(axes, x, y=None, yhat=None, layer_name='', title=None, img_value=True):
        n_images = len(axes)
        minv, maxv = np.min(x), np.max(x)
        for j, image in enumerate(x[:n_images]):
            ax = axes[j]
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

            ax.imshow(np.atleast_2d(image.squeeze()),
                      cmap='gray', vmin=minv, vmax=maxv)

            if img_value:
                image_2d = np.atleast_2d(image.squeeze())
                for (ii, jj), val in np.ndenumerate(image_2d):
                    ax.text(jj, ii, f'{val:.2f}', ha='center',
                            va='center', color='red', fontsize=8)

    def visualize_filter(self, layer_name, **kwargs):
        try:
            layer = self.model
            for name in layer_name.split('.'):
                layer = getattr(layer, name)
            if isinstance(layer, nn.Conv2d):
                weights = layer.weight.data.cpu().numpy()
                n_filters, n_channels, _, _ = weights.shape

                size = (2 * n_channels + 2, 2 * n_filters)
                fig, axes = plt.subplots(n_filters, n_channels, figsize=size)
                axes = np.atleast_2d(axes)

                for i in range(n_filters):
                    StepByStep._visualize_tensors(
                        axes[i, :],
                        weights[i],
                        layer_name=f'Filter #{i}',
                        title='Channel',
                    )

                fig.tight_layout()
                for ax in axes.flat:
                    ax.label_outer()

                return fig
        except AttributeError:
            return

    def attach_hooks(self, layers_to_hook, hook_fn=None):
        self.visualization = {}
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules}

        if hook_fn is None:
            def hook_fn(layer, inputs, outputs):
                name = layer_names[layer]
                values = outputs.detach().cpu().numpy()
                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([
                        self.visualization[name], values
                    ])

        for name, layer in modules:
            if name in layers_to_hook:
                self.visualization[name] = None
                self.handlers[name] = layer.register_forward_hook(hook_fn)

    def remove_hooks(self):
        for name, handler in self.handlers.items():
            handler.remove()
        self.handlers = {}

    def visualize_outputs(self, layers, n_images=10, y=None, y_hat=None):
        layers = [l for l in layers if l in self.visualization.keys()]
        shapes = [self.visualization[layer].shape for layer in layers]

        n_rows = []
        for shape in shapes:
            if len(shape) == 4:
                n_rows.append(shape[1])
            elif len(shape) == 2:
                n_rows.append(1)
            else:
                raise ValueError(f"Unsupported shape: {shape}")

        total_rows = np.sum(n_rows)

        fig, axes = plt.subplots(total_rows, n_images, figsize=(
            1.5 * n_images, 1.5 * total_rows))

        current_row = 0
        for i, layer in enumerate(layers):
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
        x, y = x.to(self.device), y.to(self.device)

        with torch.inference_mode():
            yhat = self.model(x)

        n_samples, n_dims = yhat.shape

        if n_dims > 1:
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims = 2

            if yhat.min() >= 0 and yhat.max() <= 1:
                predicted = (yhat > threshold).long()
            else:
                predicted = (F.sigmoid(yhat) > threshold).long()

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
