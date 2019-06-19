import numpy as np
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Callable, List, Tuple
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


def train(
        model: nn.Module,
        train_data: DataLoader,
        loss_fn: LossFn,
        optimizer: Optimizer,
        num_epochs: int,
        lr_decay: Callable[[Optimizer], None],
) -> List[float]:
    print('Train started')
    res = []
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
        lr_decay(optimizer)
        el = np.array(epoch_loss)
        mean = el.mean()
        print(
            'epoch: {} loss_mean: {} loss_max: {} loss_min: {}'
            .format(epoch, mean, el.max(), el.min())
        )
        res.append(float(mean))
    return res
