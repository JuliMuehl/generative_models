import torch
import numpy as np
import matplotlib.pyplot as plt
import time

model = torch.load("wdcgan.pt", weights_only=False)

plt.ion()
fig, ax = plt.subplots(8,8)
while True:
    grid_samples = model.sample(8*8).to("cpu").detach()
    samples_np = grid_samples.view(8, 8, *grid_samples.shape[1:]).permute([0, 1, 3, 4, 2]).numpy()
    for i in range(8):
        for j in range(8):
            ax[i][j].clear()
            ax[i][j].imshow(samples_np[i, j], aspect="auto")
            ax[i][j].axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1.0)
plt.ioff()
