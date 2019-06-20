from datetime import datetime
import numpy as np
from pathlib import Path
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Callable
from .loss import discretized_mix_logistic_loss


LossFn = Callable[[Tensor, Tensor], Tensor]


def prepare_data(dataset: str, data_dir: str, batch_size: int, train: bool = True) -> DataLoader:
    trans = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x - 0.5) * 2.0
    ])
    if dataset == 'cifar':
        data = datasets.CIFAR10(data_dir, train=train, download=True, transform=trans)
    else:
        raise ValueError('dataset {} is not supported'.format(dataset))
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def loss_fn(dataset: str) -> LossFn:
    if dataset == 'cifar':
        return discretized_mix_logistic_loss
    else:
        raise ValueError('dataset {} is not supported'.format(dataset))


def save_model(model: nn.Module, optimizer: Optimizer, log_file: str) -> None:
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(save_dict, log_file)


def train(
        model: nn.Module,
        train_data: DataLoader,
        loss_fn: LossFn,
        optimizer: Optimizer,
        num_epochs: int,
        save_freq: int,
        log_dir: str,
        lr_decay: callable,
) -> None:
    print('Train started')
    model.train(True)
    log_dir = Path(log_dir)
    if not log_dir.exists():
        log_dir.mkdir()
    loss_list = []
    start_time = datetime.now()
    for epoch in range(num_epochs):
        epoch_loss = []
        for img, _ in train_data:
            img = img.to(model.device)
            output = model(img)
            loss = loss_fn(img, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.item()))
        lr_decay()
        el = np.array(epoch_loss)
        mean = el.mean()
        if epoch > 0 and epoch % save_freq == 0:
            save_model(model, optimizer, log_dir.joinpath('model.pth.{}'.format(epoch)))
        print(
            'epoch: {} loss_mean: {} loss_max: {} loss_min: {}, elapsed: {}'
            .format(epoch, mean, el.max(), el.min(), (start_time - datetime.now()).total_seconds)
        )
        loss_list.append(float(mean))
    np.save(log_dir.joinpath('loss.npy'), np.array(loss_list))
