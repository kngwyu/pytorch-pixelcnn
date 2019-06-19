import click
from typing import Optional
from .pixelcnn_pp import PixelCNNpp
from .train_helper import prepare_data, train


@click.group()
@click.option('--num-gpu', type=int, default=1)
@click.pass_context
def main_app(ctx: dict, num_gpu: int) -> None:
    ctx['gpu'] = num_gpu


@main_app.command()
@click.option('--log-dir', type=str, default='./log')
@click.option('--data-dir', type=str, default='./data')
@click.option('--dataset', type=str, default='cifar')
@click.option('--lr', type=float, default=0.001)
@click.option('--lr-decay', type=float, default=None)
@click.option('--batch-size', type=int, default=16)
@click.option('--epochs', type=int, default=5000)
def train_app(
        ctx: dict,
        log_dir: str,
        data_dir: str,
        dataset: str,
        lr: float,
        lr_decay: Optional[float],
        batch_size: int,
        num_groups: int,
        num_layers: int,
        hidden_channel: int,
        downsize_stride: int,
        num_logistic_mix: int,
) -> None:
    data = prepare_data(dataset, data_dir, batch_size)
    pass


@main_app.command()
def test_app() -> None:
    pass


if __name__ == '__main__':
    pass
