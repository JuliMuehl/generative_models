import torch
from torchvision.datasets import CIFAR10, MNIST
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from generative_models.gan import WassersteinGANModel
from generative_models.utils import get_device

if __name__ == '__main__':
    cifar10 = CIFAR10(root = './cifar10_data', download=True)
    x = torch.tensor(cifar10.data[np.where(np.array(cifar10.targets) == 1)], dtype=torch.float).permute([0, 3, 1, 2]) / 255.0
    device = get_device()
    critic_base = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 4, 2, 1),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Conv2d(64, 128, 4, 2, 1),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Conv2d(128, 256, 4, 2, 1),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Conv2d(256, 512, 4, 2, 1),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Conv2d(512, 1024, 4, 2, 1),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Flatten(start_dim=1)
    )
    n_features = critic_base(x[:1, ...]).shape[1]
    critic = torch.nn.Sequential(critic_base, torch.nn.Linear(n_features, 1))

    generator = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(512, 512, 4, 2, 1), 
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(True),
        torch.nn.ConvTranspose2d(512, 256, 4, 2, 1), 
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(True),
        torch.nn.ConvTranspose2d(256, 128, 4, 2, 1), 
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(True),
        torch.nn.ConvTranspose2d(128, 64, 4, 2, 1), 
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(True),
        torch.nn.ConvTranspose2d(64, 3, 4, 2, 1), 
        torch.nn.Sigmoid()
    )
    latent_shape = (512, 1, 1)
    model = WassersteinGANModel(generator, critic, latent_shape=latent_shape).to(device)

    epochs, batch_size = 1000, 256
    num_iters = x.shape[0] // batch_size

    gen_optimizer, critic_optimizer = torch.optim.RMSprop(generator.parameters(), lr=1e-4), torch.optim.RMSprop(critic.parameters(), lr=1e-4)

    plt.ion()
    ngrid = 8
    ncritic = 5
    fig, ax = plt.subplots(ngrid,ngrid)
    grid_z = torch.randn(ngrid*ngrid, *latent_shape).to(device)
    for epoch in (range(epochs)):
        for i in tqdm(range(num_iters)):
            batch_idx = torch.randint(0, x.shape[0], (batch_size,))
            xbatch = x[batch_idx, ...].to(device)
            for i in range(ncritic):
                samples, generator_loss, critic_loss = model(xbatch)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
            samples, generator_loss, critic_loss = model(xbatch, compute_discriminator_loss=False)
            gen_optimizer.zero_grad()
            generator_loss.backward()
            gen_optimizer.step()
        grid_samples = generator(grid_z).detach().to("cpu")
        samples_np = grid_samples.view(ngrid, ngrid, *grid_samples.shape[1:]).permute([0, 1, 3, 4, 2]).numpy()
        for i in range(ngrid):
            for j in range(ngrid):
                ax[i][j].clear()
                ax[i][j].imshow(samples_np[i, j], aspect="auto")
                ax[i][j].axis('off')
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        torch.save(model, "wdcgan.pt")
