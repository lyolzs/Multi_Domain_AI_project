import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def forward_hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    # input 是一个元组，我们关心它的第一个元素
    input_var: torch.Tensor = input[0]
    # output 通常是一个张量，但为了安全也可以检查类型
    output_var: Optional[torch.Tensor] = output if isinstance(
        output, torch.Tensor) else None

    logger.debug(
        f"Layer '{module.__class__.__name__}': \n"
        f"  - Input shape:  {input_var.shape}\n"
        f"  - Output shape: {output_var.shape}\n"  # type: ignore
    )

# 1. 定义一个反向传播的 Hook 函数


def backward_grad_hook(module: nn.Module, grad_input: Tuple[Optional[torch.Tensor], ...], grad_output: Tuple[Optional[torch.Tensor], ...]) -> None:
    """
    一个简单的反向传播 Hook,用于打印梯度的统计信息。
    """
    # grad_output 是一个元组，我们通常关心第一个元素
    grad: Optional[torch.Tensor] = grad_output[0]

    if grad is not None:
        logger.debug(
            f"Backward hook on '{module.__class__.__name__}':\n"
            f"  - grad_output shape: {grad.shape}\n"
            f"  - grad_output mean: {grad.mean():.4f}\n"
            f"  - grad_output std:  {grad.std():.4f}\n"
        )
    else:
        logger.warning(
            f"Backward hook on '{module.__class__.__name__}': No gradient received.")
