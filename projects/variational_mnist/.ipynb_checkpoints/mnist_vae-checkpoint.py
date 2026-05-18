import torch
from torch import nn
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mnist = MNIST(root="./mnist_data", download=True)
    mnist_data = mnist.data.to(torch.float32) / 255.0
    plt.imshow(mnist_data[:, :, 0])
    plt.show()

    encoder = nn.Sequential(
    	nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding="same"),
    	nn.ReLU(),
    	nn.Conv2d(in_channels=10, out_channels=24, kernel_size=5, padding="same"),
    	nn.ReLU(),
    )
    decoder = nn.Sequential(
    	nn.ConvTranspose2d(in_channels=24, out_channels=10, kernel_size=5, padding="same"),
    	nn.ReLU(),
    	nn.ConvTranspose2d(in_channels=10, out_channels=1, kernel_size=5, padding="same"),
    	nn.Sigmoid()
    )
    