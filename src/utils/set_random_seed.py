import torch
import numpy as np
import random


def set_seed(seed: int):
    """
    为所有相关的随机数生成器设置种子，以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 警告：这可能会降低训练速度
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
