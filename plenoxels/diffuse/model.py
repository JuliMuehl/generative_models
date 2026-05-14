import torch
import torch.nn.functional as F

class DiffuseVoxelGrid(torch.nn.Module):
    def __init__(self, scale, N):
        super().__init__()
        self.scale = scale
        self.N = N
        self.color_grid = torch.nn.Parameter(torch.zeros(1, 3, N, N, N))
        self.density_grid = torch.nn.Parameter(torch.zeros(1, 1, N, N, N))

    #Assumes xyz \in (-1,1)^3 is normalized
    def sample(self, xyz):
        n = 1
        for d in xyz.shape[:-1]:
            n *= d
        xyz = xyz.view(1, 1, 1, n, 3)
        colors = F.sigmoid(F.grid_sample(self.color_grid, xyz, mode="bilinear", align_corners=False))
        densities = F.softplus(F.grid_sample(self.density_grid, xyz, mode="bilinear", align_corners=False))
        return colors.view(*xyz.shape[:-1], 3), densities.view(*xyz.shape[:-1], 1)

    def tv_loss(self):
        color_dxsq = torch.sum((self.color_grid[:, :, :-1, :, :] - self.color_grid[:, :, 1:, :, :])**2)
        color_dysq = torch.sum((self.color_grid[:, :, :, :-1, :] - self.color_grid[:, :, :, 1:,  :])**2)
        color_dzsq = torch.sum((self.color_grid[:, :, :, :, :-1] - self.color_grid[:, :, :, :, 1:])**2)
        tv_color = torch.sqrt(color_dxsq + color_dysq + color_dzsq)
        density_dxsq = torch.sum((self.density_grid[:, :, :-1, :, :] - self.density_grid[:, :, 1:, :, :])**2)
        density_dysq = torch.sum((self.density_grid[:, :, :, :-1, :] - self.density_grid[:, :, :, 1:,  :])**2)
        density_dzsq = torch.sum((self.density_grid[:, :, :, :, :-1] - self.density_grid[:, :, :, :, 1:])**2)
        tv_density = torch.sqrt(density_dxsq + density_dysq + density_dzsq)
        return tv_color + tv_density 

    def forward(self, ray_origins, ray_directions, targets=None, num_samples=32):
        t = torch.linspace(0, 2, num_samples)
        xyz = (ray_origins[None, :, None] + t[None, :, None] * ray_directions[:, None, :]) / self.scaling * 2
        colors, densities = self.sample(xyz)
        local_transmittance = torch.exp(-densities*(t[1] - t[0]))
        transmittance = torch.cumprod(local_transmittance, dim=1) * (1 - local_transmittance) / local_transmittance
        irradiance = torch.sum(transmittance * colors, dim=1) / num_samples
        if targets is not None:
            loss = F.mse_loss(irradiance,targets) + self.tv_loss()
            return irradiance, loss
        return irradiance

if __name__ == "__main__":
    grid = DiffuseVoxelGrid((1,1,1), 64)
    origin = torch.zeros(10, 3)
    direction = torch.ones(10, 3) * 3**(-0.5)
    irradiance = grid.forward(origin, direction)
    print(irradiance.shape)
