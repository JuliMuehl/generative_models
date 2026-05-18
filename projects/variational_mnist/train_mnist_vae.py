import torch
from torch import nn
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from variational_autoencoder import VariationalAutoencoder
from tqdm import tqdm
import sys

if __name__ == "__main__":
    mnist = MNIST(root="./mnist_data", download=True)
    mnist_data = mnist.data.to(torch.float32) / 255.0
    plt.imshow(mnist_data[0, :, :])
    plt.show()

    device = "xpu" if torch.xpu.is_available() else "cpu"
    if len(sys.argv) > 1 and sys.argv[1].lower().startswith("resume"):
        print("Loading VAE")
        vae = torch.load("./mnist_vae.pt", weights_only=False)
    else:
        encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
        )
        decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5),
            nn.Sigmoid()
        )
        latent_shape = encoder(mnist_data[:1, None, :, :]).shape[1:]
        vae = VariationalAutoencoder(encoder, decoder, latent_shape).to(device)
    epochs, batch_size, lr = 5000, 64, 1e-3
    optimizer = torch.optim.AdamW(vae.parameters(),lr=lr)
    pbar = tqdm(range(epochs))
    for i in pbar:
        optimizer.zero_grad()
        batch_idx = torch.randint(mnist_data.shape[0], (batch_size,))
        x_batch = mnist_data[batch_idx, None, ...].to(device)
        _, loss = vae.forward(x_batch)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss = {loss.item()}")
        if i % 100 == 0:
            torch.save(vae, "./mnist_vae.pt")

    vae.sample()
