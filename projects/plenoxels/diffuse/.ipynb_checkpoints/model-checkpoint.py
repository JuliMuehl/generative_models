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

    def forward(self, ray_origins, ray_directions, num_samples=32):
        t = torch.linspace(0, 2, num_samples)
        xyz = (ray_origins[None, :, None] + t[None, :, None] * ray_directions[:, None, :]) / self.scaling * 2
        colors, densities = self.sample(xyz)
        local_transmittance = torch.exp(-densities*(t[1] - t[0]))
        transmittance = torch.cumprod(local_transmittance, dim=1) * (1 - local_transmittance) / local_transmittance
        return torch.sum(transmittance * colors, dim=1) / num_samples

if __name__ == "__main__":
    grid = DiffuseVoxelGrid((1,1,1), 64)
    origin = torch.zeros(10, 3)
    direction = torch.ones(10, 3) * 3**(-0.5)
    grid.forward(origin, direction)

