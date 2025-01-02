import numpy as np
import torch
from torch.utils.data import random_split, Dataset, WeightedRandomSampler
import matplotlib.pyplot as plt


class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)


def index_splitter(n, splits, seed=13):
    """将n:数量, splits:切分数, seed:随机种子"""
    idx = torch.arange(n)

    split_tensor = torch.as_tensor(splits)
    multiplier = n // split_tensor.sum()
    split_tensor = (multiplier * split_tensor).long()

    diff = n - split_tensor.sum()
    split_tensor[0] += diff

    torch.manual_seed(seed)
    return random_split(idx, split_tensor)


def make_balanced_sampler(y):
    y_tensor = torch.as_tensor(y)

    classes, count = y_tensor.unique(return_counts=True)
    weights = 1. / count.float()

    # classes_indices = {c: i for i, c in enumerate(classes)}
    # y_weights_indices = [classes_indices(label) for label in y]
    y_weights_indices = torch.searchsorted(classes, y_tensor)

    sample_weights = weights[y_weights_indices]

    generator = torch.Generator()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y),
        replacement=True,
        generator=generator,
    )


def plot_images(images, labels, n_plots=30, n_col=6):
    # 防止n_plots大于images的数量
    n_plots = min(n_plots, len(images))

    n_rows = n_plots // n_col + (n_plots % n_col > 0)
    cols = min(1.5 * n_col, 15)

    fig, axes = plt.subplots(
        n_rows, n_col, figsize=(cols, 1.5 * n_rows))
    axes = np.atleast_2d(axes)

    for i, (image, label) in enumerate(zip(images[:n_plots], labels[:n_plots])):
        ax = axes[i // n_col, i % n_col]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"#{i} - {label}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()

    plt.tight_layout()
    return fig


class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]

        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[index]

    def __len__(self):
        return len(self.x)
