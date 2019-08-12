import click
import torch
from torch import optim
from torchvision.utils import save_image
from .pixelcnn_pp import PixelCNNpp
from . import initialize
from .sample import sample_image
from . import train_helper


def kwargs_to_click_opt(f):
    annon = f.__annotations__
    for name, value in zip(annon.keys(), f.__defaults__):
        click.option('--' + name.replace('_', '-'), type=annon[name], default=value)(f)
    return f


@click.group()
def cli() -> None:
    pass


@cli.command()
@kwargs_to_click_opt
def train(
        log_dir: str = './log',
        data_dir: str = './data',
        dataset: str = 'cifar',
        lr: float = 0.001,
        lr_decay: float = 0.999995,
        batch_size: int = 16,
        epochs: int = 500,
        num_groups: int = 3,
        num_layers: int = 5,
        hidden_channel: int = 80,
        downsize_stride: int = 2,
        num_logistic_mix: int = 10,
        save_freq: int = 100,
) -> None:
    data = train_helper.prepare_data(dataset, data_dir, batch_size)
    input_channel = data.dataset.data[0].shape[2]
    model = PixelCNNpp(
        input_channel,
        num_groups,
        num_layers,
        hidden_channel,
        downsize_stride,
        num_logistic_mix,
    )
    initialize.orthogonal()(model)
    adam = optim.Adam(model.parameters(), lr=lr,)
    scheduler = optim.lr_scheduler.StepLR(adam, step_size=1, gamma=lr_decay)
    train_helper.train(
        model,
        data,
        train_helper.loss_fn(dataset),
        adam,
        epochs,
        save_freq,
        log_dir,
        lambda: scheduler.step(),
    )


@cli.command()
@kwargs_to_click_opt
def sample(
        path: str = './log/model.pth',
        data_dir: str = './data',
        dataset: str = 'cifar',
        input_channel: int = 3,
        num_groups: int = 3,
        num_layers: int = 5,
        hidden_channel: int = 80,
        downsize_stride: int = 2,
        num_logistic_mix: int = 10,
        sample_batch_size: int = 25,
) -> None:
    data = train_helper.load_dataset(dataset, data_dir)
    shape = data.data[0].shape
    model = PixelCNNpp(
        shape[2],
        num_groups,
        num_layers,
        hidden_channel,
        downsize_stride,
        num_logistic_mix,
    )
    data = torch.load(path)
    model.load_state_dict(data['model'])
    out = sample_image(model, sample_batch_size, shape[2], shape[0], shape[1], num_logistic_mix)
    out = out * 0.5 + 0.5
    save_image(out, 'image.png', nrow=5, padding=0)


if __name__ == '__main__':
    cli()
