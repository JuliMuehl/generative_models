import torch
import torch.nn.functional as F

class GaussianPosteriorLayer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mean_layer = torch.nn.Linear(dim, dim, bias=True)
        self.sigma_layer= torch.nn.Linear(dim, 1, bias=True)
        self.dim = dim

    def forward(self, x):
        mu = self.mean_layer(x)
        sigma = F.relu(self.sigma_layer(x))
        kl_loss = 0.5 * (self.dim * torch.sum(sigma**2) + torch.sum(mu**2) - 2 * self.dim * torch.sum(torch.log(sigma)))
        return mu, sigma, kl_loss


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, encoder, decoder, latent_shape):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_shape = latent_shape
        self.latent_dim = 1 
        for i in latent_shape:
            self.latent_dim *= i
        self.posterior_layer = GaussianPosteriorLayer(self.latent_dim)

    def forward(self, x, num_samples=16):
        device = next(self.parameters()).device
        input_shape = x.shape[1:]
        batch_size = x.shape[0]
        e = self.encoder(x)
        mu, sigma, kl_loss = self.posterior_layer(e.view([-1, self.latent_dim]))
        rng = torch.randn((batch_size, num_samples, self.latent_dim)).to(device)
        z_samples = sigma[:, None] * rng  + mu[:, None, :]
        x_samples = self.decoder(z_samples.view([batch_size * num_samples, *self.latent_shape]))
        x_samples = x_samples.view([batch_size, num_samples, *input_shape])
        log_loss = 0.5 * torch.sum((x_samples - x[:, None, ...])**2)
        return x_samples, log_loss + kl_loss

    def sample(self, num_samples=16):
        device = next(self.parameters()).device
        z_samples = torch.randn((num_samples, self.latent_dim)).to(device)
        x_samples = self.decoder(z_samples.view([num_samples, *self.latent_shape]))
        return x_samples
