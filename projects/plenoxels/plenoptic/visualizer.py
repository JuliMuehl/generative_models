from generative_models.plenoxels.renderers import GroundTruthRenderer, SH2VoxelRenderer
from generative_models.plenoxels.renderers.utils import *

import moderngl
import numpy as np
import glfw
import time
import os

def main():
    glfw.init()
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    WINDOW_WIDTH, WINDOW_HEIGHT = 1024, 512
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, __file__, None, None)
    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    voxel_renderer = SH2VoxelRenderer(ctx=ctx)
    groundtruth_renderer = GroundTruthRenderer(ctx=ctx)
    phi, theta = 0.0, 2*np.pi/5
    camera_pos = unit_sphere(theta, phi)
    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)
        ctx.clear(0.0,0.0,0.0,1.0)
        mouse_x, mouse_y = glfw.get_cursor_pos(window)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            phi = max(0, min(2, (mouse_x/WINDOW_WIDTH)  * 2)) * np.pi
            theta = max(0.1, min(0.4,(0.1 + (1.0 - mouse_y/WINDOW_HEIGHT) * 0.4))) * np.pi
            camera_pos = unit_sphere(theta, phi)
        voxel_renderer.set_viewport(0, 0, 512, 512)
        voxel_renderer.render_to_screen(x = camera_pos)
        groundtruth_renderer.set_viewport(512, 0, 512, 512)
        groundtruth_renderer.render_to_screen(x = camera_pos)
        glfw.poll_events()
        voxel_renderer.load_grid()

if __name__ == "__main__":
    main()
