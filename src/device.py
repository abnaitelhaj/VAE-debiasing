"""Select CUDA when available."""

import torch
import torch.backends.cudnn as cudnn


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.benchmark = True
    else:
        print("GPU not available — using CPU (training will be slow).")
        device = torch.device("cpu")
    return device
