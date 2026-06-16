This repository serves as an educational implementation of various probabalistic and 3D vision based generative models in PyTorch.

## Projects
* [Plenoxels (Neural Radiance Fields without Neural Networks)](projects/plenoxels)
* [Wasserstein GAN on CIFAR10 Automobile Class](projects/wdcgan_cifar10)
* [1D Training Visualization for Wasserstein- and Jensen-Shannon GAN](projects/gan_example)
* [Variational Autoencoders](projects/variational_mnist)

## Installation
To simplify installing the necessary dependencies it is recommended to first create a new virtualenv

```bash
python -m virtualenv venv
```

Install the requirements by running 

```bash
pip install -r requirements.txt 
```

or

```bash
pip install -r requirements_xpu.txt 
```

if you are using an intel gpu/xpu.

You can now install the main package from the *src/* directory by running

```bash
pip install -e .
```
