# PyTorch-PixelCNN
Currently this repository contains an implementation of [PixelCNN++](https://arxiv.org/abs/1606.05328).

I will add [PixelCNN](https://arxiv.org/abs/1606.05328) and other varitants.

## Sampled image
![Sampled image](./pictures/image.png)

## Setup
First, install [pipenv](https://pipenv.readthedocs.io/en/latest/).
E.g. you can install it via
``` bash
pip install pipenv --user
```

Then you can create a virtual environment for isolated installing of related packages.
```bash
pipenv --site-packages --three install
```

**NOTE**

Pipenv installs the latest PyTorch, so if you want to use the version installed in
your machine, please specify the version in Pipfile.

E.g., if you want to use PyTorch 1.1.0,
```toml
torch = "1.1.0"
```

## Example usages

### Train the model using CIFAR-10
```bash
pipenv run python -m pixelcnn train
```

### Sample images by trained model
```bash
pipenv run python -m pixelcnn sample --path=log/model.pth.400
```

## License
This project is licensed under Apache License, Version 2.0
([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).
