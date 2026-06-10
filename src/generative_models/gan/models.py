import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class AbstractGANModel(ABC, torch.nn.Module):
    def __init__(self, generator, discriminator, latent_shape):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_shape = latent_shape
    
    @abstractmethod
    def calculate_losses(self, x, x_):
        pass

    def sample(self, num_samples):
        device = next(self.parameters()).device
        z = torch.randn((num_samples, ) + self.latent_shape).to(device)
        return self.generator(z)
    
    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        x_ = self.sample(batch_size)
        generator_loss, discriminator_loss = self.calculate_losses(x, x_, **kwargs)
        return x_, generator_loss, discriminator_loss

class JensenShannonGAN(AbstractGANModel):
    def __init__(self, generator, discriminator, latent_shape):
        super().__init__(generator, discriminator, latent_shape)

    def calculate_losses(self, x, x_, compute_discriminator_loss=True):
        y = self.discriminator(x)
        y_D = self.discriminator(x_.detach())
        y_G = self.discriminator(x_)

        d_1_labels = torch.ones_like(y)
        d_0_labels = torch.zeros_like(y_G)
        g_1_labels = torch.ones_like(y_G)
        generator_loss = F.binary_cross_entropy_with_logits(input=y_G, target=g_1_labels)
        discriminator_loss = None
        if compute_discriminator_loss:
            discriminator_loss_real = F.binary_cross_entropy_with_logits(input=y, target=d_1_labels)
            discriminator_loss_fake = F.binary_cross_entropy_with_logits(input=y_D, target=d_0_labels)
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
        return generator_loss, discriminator_loss

class WassersteinGANModel(AbstractGANModel):
    def __init__(self, generator, discriminator, latent_shape, gradient_penalty_coeff=1.0):
        super().__init__(generator, discriminator, latent_shape)
        self.gradient_penalty_coeff = gradient_penalty_coeff
    
    def gradient_penalty(self, x, x_):
        device = next(self.parameters()).device
        alpha = torch.rand((x.shape[0],) + tuple(1 for _ in x.shape[1:]), device=device)
        interp = (alpha * x + (1 - alpha) * x_).requires_grad_(True)
        d_interp = self.discriminator(interp)
        grad_output = torch.ones_like(d_interp, requires_grad=False, device=device)
        gradients = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=grad_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
        return gradient_penalty

    def calculate_losses(self, x, x_, compute_generator_loss = True, compute_discriminator_loss=True, compute_gradient_penalty = True):
        y = self.discriminator(x)
        y_D = self.discriminator(x_.detach())
        y_G = self.discriminator(x_)
        generator_loss, discriminator_loss = None, None
        if compute_generator_loss:
            generator_loss = -torch.mean(y_G)
        if compute_discriminator_loss:
            discriminator_loss = torch.mean(y_D) - torch.mean(y)
        if compute_discriminator_loss and compute_gradient_penalty:
            discriminator_loss += self.gradient_penalty_coeff * self.gradient_penalty(x, x_)
        return generator_loss, discriminator_loss
    

