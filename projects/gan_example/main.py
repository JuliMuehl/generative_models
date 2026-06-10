import torch
import numpy as np
import matplotlib.pyplot as plt

from generative_models.utils import get_device
from generative_models.gan import JensenShannonGAN, WassersteinGANModel

def rejection_sampling(x, y):
    n = len(x)
    s = np.max(y)
    i = np.random.randint(0, n)
    while np.random.random() * s > y[i]:
        i = np.random.randint(0, n)
    return x[i]


if __name__ == '__main__':
    x = np.linspace(0, 1, 1024)
    y = (x - 0.5)**2
    norm = np.mean(y)

    generator = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(32,16),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(16,8),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(8, 1),
        torch.nn.Sigmoid()
    )
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    )
    device = get_device()
    model = WassersteinGANModel(generator, discriminator, latent_shape=(1,), gradient_penalty_coeff=0.01).to(device)
    gen_optimizer = torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    discr_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    num_iters, batch_size, ncritic = 5000, 512, 5
    plt.ion()
    fig, ax = plt.subplots(3)
    gen_losses, discr_losses = [], []
    y_density = y/np.mean(y)
    xgpu = torch.tensor(x, dtype=torch.float).to(device)[:, None]
    for _ in range(num_iters):
        xbatch = torch.tensor([rejection_sampling(x, y) for _ in range(batch_size)])[:, None].to(torch.float).to(device)
        for i in range(ncritic):
            samples, generator_loss, discr_loss = model(xbatch, compute_generator_loss=False)
            discr_optimizer.zero_grad()
            discr_loss.backward()
            discr_optimizer.step()
        samples, generator_loss, discr_loss = model(xbatch)
        gen_optimizer.zero_grad()
        generator_loss.backward()
        gen_optimizer.step()
        gen_losses.append(generator_loss.detach().item())
        discr_losses.append(discr_loss.detach().item())
        ax[1].clear()
        ax[1].plot(discr_losses)
        ax[1].plot(gen_losses)
        ax[1].legend(["discr_losses", "generator_losses"])
        ax[1].set_title("Losses")
        ax[0].clear()
        ax[0].plot(x, y_density)
        ax[0].set_ylim(y_density.min(), y_density.max())
        ax[0].hist(samples.detach().to("cpu").numpy(), bins=16, density=True, stacked=True)
        ax[0].set_title( "Generated Distribution")
        f = discriminator(xgpu).detach().to("cpu").numpy()
        ax[2].clear()
        ax[2].plot(x, f)
        ax[2].set_title("Critic function")
        fig.canvas.draw()
        fig.canvas.flush_events()    

