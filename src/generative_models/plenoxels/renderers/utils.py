import numpy as np

def tangent_frame(x, up, dtype=np.float32):
        res = np.eye(3, dtype=dtype)
        res[:, 2] = -np.array(x, dtype=dtype)
        res[:, 0] = np.cross(res[:, 2], up)
        res[:, 1] = np.cross(res[:, 2], res[:, 0])
        return res / np.linalg.norm(res, axis=0)

def unit_sphere(theta, phi, dtype=np.float32):
    return [
        np.sin(theta) * np.cos(phi),
        np.cos(theta),
        np.sin(theta) * np.sin(phi),
    ]
