import torch
import torch.nn.functional as F

class DiffuseVoxelGrid(torch.nn.Module):
    def __init__(self, scale, N, tv_loss_weight=1e-5, sparsity_loss_weight = 1e-2, bg_rgb = [1.0, 1.0, 1.0]):
        super().__init__()
        self.scale = scale
        self.N = N
        self.color_grid = torch.nn.Parameter(0.2 * torch.randn(1, 3, N, N, N))
        self.density_grid = torch.nn.Parameter(-20 * torch.ones(1, 1, N, N, N))
        self.tv_loss_weight = tv_loss_weight
        self.sparsity_loss_weight = sparsity_loss_weight
        self.register_buffer("bg_rgb", torch.tensor(bg_rgb, dtype=torch.float))

    def sample(self, xyz):
        n = 1
        shape = xyz.shape
        for d in xyz.shape[:-1]:
            n *= d
        xyz = xyz.view(1, 1, 1, n, 3)
        colors = F.sigmoid(F.grid_sample(self.color_grid, xyz, mode="bilinear", align_corners=True))
        densities =  F.softplus(F.grid_sample(self.density_grid, xyz, mode="bilinear", align_corners=True))
        return colors.view(3, *shape[:-1]).permute(1,2,0), densities.view(1, *shape[:-1]).permute(1,2,0)

    def tv_loss(self):
        color_dxsq = (self.color_grid[:, :, :-1, :-1, :-1] - self.color_grid[:, :,  1:, :-1,:-1])**2
        color_dysq = (self.color_grid[:, :, :-1, :-1, :-1] - self.color_grid[:, :, :-1,  1:,:-1])**2
        color_dzsq = (self.color_grid[:, :, :-1, :-1, :-1] - self.color_grid[:, :, :-1, :-1, 1:])**2
        tv_color = torch.mean(torch.sqrt(color_dxsq + color_dysq + color_dzsq + 1e-5))
        density_dxsq = (self.density_grid[:, :, :-1, :-1, :-1] - self.density_grid[:, :,  1:, :-1, :-1])**2
        density_dysq = (self.density_grid[:, :, :-1, :-1, :-1] - self.density_grid[:, :, :-1,  1:, :-1])**2
        density_dzsq = (self.density_grid[:, :, :-1, :-1, :-1] - self.density_grid[:, :, :-1, :-1,  1:])**2
        tv_density = torch.mean(torch.sqrt(density_dxsq + density_dysq + density_dzsq + 1e-5))
        return  (tv_color + tv_density)

    def sparsity_loss(self):
        sigma = F.softplus(self.density_grid)
        density_loss = torch.mean(torch.log(1 + 2*sigma**2))
        c = F.sigmoid(self.color_grid)
        color_loss = torch.mean(torch.log(1 + 2 * c**2))
        return density_loss + color_loss

    def forward(self, ray_origins, ray_directions, targets=None, num_samples=32, calc_tv_loss = False):
        device = next(self.parameters()).device
        t = torch.linspace(0, 2.0, num_samples).to(device)
        xyz = ray_origins[:, None, :] / self.scale + t[None, :, None] * ray_directions[:, None, :]
        colors, densities = self.sample(xyz)
        alpha = 1.0 - torch.exp(-densities*(t[1] - t[0]))
        transmittance = torch.cumprod(1.0 - alpha, dim=1) 
        weights = alpha * transmittance
        background = transmittance[:, -1:, 0] * self.bg_rgb[None, :]
        radiance = torch.sum(weights * colors, dim=1) #+ background
        if targets is not None:
            loss = F.mse_loss(radiance,targets) + self.sparsity_loss_weight * self.sparsity_loss()
            if calc_tv_loss:
                loss += self.tv_loss_weight * self.tv_loss()
            return radiance, loss
        return radiance 
