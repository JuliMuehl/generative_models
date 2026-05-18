import torch
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image

from generative_models.plenoxels.diffuse import DiffuseVoxelGrid
from generative_models.utils import get_device

if __name__ == "__main__":
    device = get_device()
    model = DiffuseVoxelGrid(scale=1, N=64, tv_loss_weight=1e-4, sparsity_loss_weight=1e-4).to(device)
    image_dir = "train_images"
    with open(os.path.join(image_dir, "camera_poses.json")) as f:
        camera_positions = np.array(json.load(f)["camera_positions"])
    num_poses = len(camera_positions)
    colors = [] 
    dirs = []
    for i in tqdm(range(num_poses)):
        with Image.open(os.path.join(image_dir, f"img{i}.png")) as img:
            img = np.array(img).reshape(-1, len(img.getbands())) / 255.0
            colors.append(img[:, :3])
        with Image.open(os.path.join(image_dir, f"dirs{i}.png")) as img:
            img_dirs = np.array(img) / 255.0 * 2.0 - 1.0
            img_dirs = img_dirs.reshape(-1, 3)
            dirs.append(img_dirs)
    
    #Due to the 8bit quantization of the png file the directions aren't quite normalized anymore so we renormalize them here
    dirs = np.array(dirs)
    dirs = dirs/np.linalg.norm(np.array(dirs), axis=-1)[:, :, None]
    origins = np.repeat(camera_positions[:, None, :], dirs.shape[1], axis=1)
    #converting to np.array from list[np.array] first instead of directly converting to torch.tensor improves speed
    colors = torch.tensor(np.array(colors), dtype=torch.float).to(device)
    dirs   = torch.tensor(dirs, dtype=torch.float).to(device)
    origins = torch.tensor(origins, dtype=torch.float).to(device)

    optimizer = torch.optim.AdamW([{'params':model.color_grid, 'lr' : 1e-1}, {'params':model.density_grid, 'lr':3.0}])
    n_splits = 512 
    epochs, batch_size = 1000, len(colors.view(-1,3)) // n_splits
    pbar = tqdm(range(epochs))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-2 ** (1.0 / epochs))
    for epoch in pbar:
        batch_idx = torch.randint(colors.view(-1, 3).shape[0], (batch_size,)).to(device)
        batch_colors = colors.view(-1, 3)[batch_idx, :]
        batch_dirs = dirs.view(-1, 3)[batch_idx, :]
        batch_origins = origins.view(-1, 3)[batch_idx, :]
        optimizer.zero_grad()
        _, loss = model.forward(batch_origins, batch_dirs, targets=batch_colors, calc_tv_loss = True)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss = {loss.item()}")
        scheduler.step()
        if epoch % 10 == 0:
            model.save_numpy()
