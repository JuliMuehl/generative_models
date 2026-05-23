import torch

def get_device():
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_avaialble():
        return "cuda"
    return "cpu"