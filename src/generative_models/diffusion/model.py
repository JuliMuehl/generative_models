import torch

class DiffusionModel(torch.nn.Module):
    def __init__(self, input_shape, denoiser, beta_start=1e-4, beta_end=0.012, num_steps=1000):
        super().__init__()
        self.input_shape = input_shape
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        self.denoiser = denoiser
        self.register_buffer("self.beta", torch.linspace(self.beta_start, self.beta_end, self.num_steps))
        self.alpha = 1.0 - self.beta
        self.calpha = torch.cumprod(self.alpha, dim=-1)
        self.sigma = torch.sqrt(1.0 - self.calpha)
    
    def forward(self, x):
        pass
