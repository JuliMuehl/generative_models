import torch
import torch.nn.functional as F
import numpy as np
import os

class DiffuseVoxelGrid(torch.nn.Module):
    def __init__(self, scale, N, tv_loss_weight=1e-3, sparsity_loss_weight = 1.0, bg_rgb = [1.0, 1.0, 1.0], softplus_beta = 1e1):
        super().__init__()
        self.scale = scale
        self.N = N
        self.color_grid = torch.nn.Parameter(0.2 * torch.randn(1, 3, N, N, N))
        self.density_grid = torch.nn.Parameter(-1.0 * torch.ones(1, 1, N, N, N))
        self.tv_loss_weight = tv_loss_weight
        self.sparsity_loss_weight = sparsity_loss_weight
        self.softplus_beta = softplus_beta
        self.register_buffer("bg_rgb", torch.tensor(bg_rgb, dtype=torch.float))

    def softplus(self, x):
        return F.softplus(self.softplus_beta * x) / self.softplus_beta

    def sample(self, xyz):
        n = 1
        shape = xyz.shape
        for d in xyz.shape[:-1]:
            n *= d
        xyz = xyz.view(1, 1, 1, n, 3)
        colors = F.sigmoid(F.grid_sample(self.color_grid, xyz, mode="bilinear", align_corners=True))
        densities =  self.softplus(F.grid_sample(self.density_grid, xyz, mode="bilinear", align_corners=True))
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

    def forward(self, ray_origins, ray_directions, targets=None, num_samples=32, calc_tv_loss = False):
        device = next(self.parameters()).device
        t = torch.linspace(0, 2.0, num_samples).to(device)
        xyz = ray_origins[:, None, :] / self.scale + t[None, :, None] * ray_directions[:, None, :]
        colors, densities = self.sample(xyz)
        alpha = 1.0 - torch.exp(-densities*(t[1] - t[0]))
        transmittance = torch.cumprod(1.0 - alpha, dim=1) 
        weights = alpha * transmittance
        background = transmittance[:, -1:, 0] * self.bg_rgb[None, :]
        radiance = torch.sum(weights * colors, dim=1) + background
        if targets is not None:
            sparsity_loss = torch.mean(torch.log(1 + 4 * alpha * (1 - alpha)))
            loss = F.mse_loss(radiance,targets) + self.sparsity_loss_weight * sparsity_loss
            if calc_tv_loss:
                loss += self.tv_loss_weight * self.tv_loss()
            return radiance, loss
        return radiance 

    def save_numpy(self):
        color_grid = F.sigmoid(self.color_grid).view(3, self.N, self.N, self.N).permute(1,2,3,0).detach().to("cpu").numpy()
        density_grid = self.softplus(self.density_grid).view(1, self.N, self.N, self.N).permute(1, 2, 3, 0).detach().to("cpu").numpy()
        np.savez_compressed("voxel_data.part", color_grid = color_grid, density_grid = density_grid)
        os.replace("voxel_data.part.npz", "voxel_data.npz")


class SH2VoxelGrid(torch.nn.Module):
    def __init__(self, scale, N, tv_loss_weight=1e-3, sparsity_loss_weight = 1.0, bg_rgb = [1.0, 1.0, 1.0], softplus_beta = 1e1):
        super().__init__()
        self.scale = scale
        self.N = N
        self.density_grid = torch.nn.Parameter(-1.0 * torch.ones(1, 1, N, N, N))
        pi = np.pi
        self.register_buffer("sh2_C", 0.5 * torch.tensor([
            1/pi,
            3/pi, 3/pi, 3/pi,
            15/pi, 15/pi, 15/pi, 0.25*5/pi, 0.25*15/pi,
        ]).sqrt())
        self.sh2_grid = torch.nn.Parameter(0.1 * torch.ones(1, 3*9, N, N, N))
        self.tv_loss_weight = tv_loss_weight
        self.sparsity_loss_weight = sparsity_loss_weight
        self.softplus_beta = softplus_beta
        self.register_buffer("bg_rgb", torch.tensor(bg_rgb, dtype=torch.float))
        pi = np.pi

    def eval_sh2(self, dirs, coeffs):
        x, y, z = dirs[:, 0][:, None], dirs[:, 1][:, None], dirs[:, 2][:, None]
        C = self.sh2_C
        sh_basis = torch.cat([
            torch.ones_like(x),
            x,y,z,
            x*y, y*z, z*x, (3 * z*z - 1.0), (x*x - y*y)
        ], dim = -1)
        sh_basis = C[None, :] * sh_basis
        #shape of coeffs: [num_rays, num_samples, 3, 9]
        color = torch.relu(torch.sum(coeffs * sh_basis[:, None, None, :], dim=-1))
        return color

    def softplus(self, x):
        return F.softplus(self.softplus_beta * x) / self.softplus_beta

    def sample(self, xyz, dirs):
        shape = xyz.shape
        n = 1
        for d in xyz.shape[:-1]:
            n *= d
        xyz = xyz.view(1, 1, 1, n, 3)
        sh2_coeffs = F.grid_sample(self.sh2_grid, xyz, mode="bilinear", align_corners=True)
        densities =  self.softplus(F.grid_sample(self.density_grid, xyz, mode="bilinear", align_corners=True))
        sh2_coeffs = sh2_coeffs.view(3, 9, *shape[:-1]).permute(2,3,0,1)
        colors = self.eval_sh2(dirs, sh2_coeffs)
        return colors, densities.view(1, *shape[:-1]).permute(1,2,0)

    def tv_loss(self):
        sh2_dxsq = (self.sh2_grid[:, :, :-1, :-1, :-1] - self.sh2_grid[:, :,  1:, :-1,:-1])**2
        sh2_dysq = (self.sh2_grid[:, :, :-1, :-1, :-1] - self.sh2_grid[:, :, :-1,  1:,:-1])**2
        sh2_dzsq = (self.sh2_grid[:, :, :-1, :-1, :-1] - self.sh2_grid[:, :, :-1, :-1, 1:])**2
        tv_sh2 = torch.mean(torch.sqrt(sh2_dxsq + sh2_dysq + sh2_dzsq + 1e-5))
        density_dxsq = (self.density_grid[:, :, :-1, :-1, :-1] - self.density_grid[:, :,  1:, :-1, :-1])**2
        density_dysq = (self.density_grid[:, :, :-1, :-1, :-1] - self.density_grid[:, :, :-1,  1:, :-1])**2
        density_dzsq = (self.density_grid[:, :, :-1, :-1, :-1] - self.density_grid[:, :, :-1, :-1,  1:])**2
        tv_density = torch.mean(torch.sqrt(density_dxsq + density_dysq + density_dzsq + 1e-5))
        return  (tv_sh2 + tv_density)

    def forward(self, ray_origins, ray_directions, targets=None, num_samples=32, calc_tv_loss = False):
        device = next(self.parameters()).device
        t = torch.linspace(0, 2.0, num_samples).to(device)
        # [num_rays , num_samples , 3]
        xyz = ray_origins[:, None, :] / self.scale + t[None, :, None] * ray_directions[:, None, :]
        colors, densities = self.sample(xyz, ray_directions)
        alpha = 1.0 - torch.exp(-densities*(t[1] - t[0]))
        transmittance = torch.cumprod(1.0 - alpha, dim=1)
        weights = alpha * transmittance
        background = transmittance[:, -1:, 0] * self.bg_rgb[None, :]
        radiance = torch.sum(weights * colors, dim=1) + background
        if targets is not None:
            sparsity_loss = torch.mean(torch.log(1 + 4 * alpha * (1 - alpha)))
            loss = F.mse_loss(radiance,targets) + self.sparsity_loss_weight * sparsity_loss
            if calc_tv_loss:
                loss += self.tv_loss_weight * self.tv_loss()
            return radiance, loss
        return radiance

    def save_numpy(self):
        sh2_grid = self.sh2_grid.view(3, 9, self.N, self.N, self.N).permute(1,2,3,4,0).detach().to("cpu").numpy()
        density_grid = self.softplus(self.density_grid).view(1, self.N, self.N, self.N).permute(1, 2, 3, 0).detach().to("cpu").numpy()
        sh2_C = self.sh2_C.detach().to("cpu").numpy()
        np.savez_compressed("voxel_data.part", sh2_C = sh2_C, sh2_grid = sh2_grid, density_grid = density_grid)
        os.replace("voxel_data.part.npz", "voxel_data.npz")
