from generative_models.plenoxels.renderers import GroundTruthRenderer
from generative_models.plenoxels.renderers.utils import *

import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

def render_and_save(renderer, image_dir, positions):
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    for i, pos in tqdm(list(enumerate(positions))):
        colors, directions= renderer.render_to_texture(x = pos)
        #flip textures to get images
        colors, directions = colors[::-1, :, :], directions[::-1, :, :]
        colors, directions= Image.fromarray(colors), Image.fromarray(directions)
        colors.save(os.path.join(image_dir, f"img{i}.png"))
        directions.save(os.path.join(image_dir,f"dirs{i}.png"))
    with open(os.path.join(image_dir, "camera_poses.json"), "w") as f:
        json.dump({"camera_positions":positions},f)

def main():    
    renderer = GroundTruthRenderer()
    train_positions = [unit_sphere(theta,phi) for theta in np.linspace(np.pi/8, np.pi/3, 8) for phi in np.linspace(0, 2 * np.pi, 8)]
    train_image_dir = "train_images"
    print("Generating Training Images:")
    render_and_save(renderer, train_image_dir, train_positions)

if __name__ == "__main__":
    main()
