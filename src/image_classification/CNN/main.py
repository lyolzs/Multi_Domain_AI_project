import torch
import torch.nn as nn
import torch.optim as optim
import logging  # 导入 logging 模块
import yaml
import sys
import os
import json
import time
import shutil
from typing import TypedDict, List, Tuple
# 使用相对导入，从同一目录下的其他模块导入函数和类
# 导入 setup_logging
# 导入 load_config 和 parse_args
from src.utils.logging import def_setup_logging
from src.utils.config import load_config, parse_args
from src.utils.set_random_seed import set_seed
from src.image_classification.CNN.model import CNN
from src.image_classification.CNN.data_loader import get_data_loaders
from src.image_classification.CNN.train import train
from src.image_classification.CNN.hooks import forward_hook, backward_grad_hook
from src.utils.visualization import plot_training_history
from src.utils.experiment import setup_train_environment

logger = logging.getLogger(__name__)  # 获取当前模块的 logger


def main():
    # 1. 加载配置和参数
    args = parse_args()
    # 2. 设置实验环境 (所有设置逻辑都被封装在这里)
    config, device, experiment_output_dir = setup_train_environment(args)

    # 4. 加载数据
    train_loader, val_loader, _ = get_data_loaders(config)

    # 5. 初始化模型、损失函数和优化器
    model = CNN(config).to(device)
    # 根据配置决定是否注册 Hooks
    if config.get('debug', {}).get('enable_hooks', False):
        print("正在为模型注册调试 Hooks...")

        # 定义一个用于 apply 的函数，使其更具可读性

        def selective_forward_hook_register(module: nn.Module):
            # 只在卷积层和最大池化层上注册 Hook
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
                module.register_forward_hook(forward_hook)

        def selective_backward_hook_register(module: nn.Module):
            # 只在卷积层和最大池化层上注册 Hook
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
                module.register_full_backward_hook(
                    backward_grad_hook)  # type: ignore

        # 使用 apply 方法将上面的函数应用到所有子模块
        model.features.apply(selective_forward_hook_register)
        model.features.apply(selective_backward_hook_register)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    logger.info("Model, criterion, and optimizer initialized.")

    # 这个检查现在变得至关重要
    if val_loader is None:
        logger.error("训练过程需要一个验证集，但未能获取。")
        sys.exit(1)  # 程序终止

    # 直接调用 train 函数即可启动整个训练过程
    training_history = train(model=model, device=device, train_loader=train_loader,
                             val_loader=val_loader, optimizer=optimizer, criterion=criterion,  config=config, output_dir=experiment_output_dir)

    logger.info("Training finished.")

    # 7. 保存并可视化结果
    # 从配置中获取保存路径，如果未定义则提供默认值

    # 7.1 保存训练历史为 JSON 文件
    history_path = os.path.join(experiment_output_dir, 'training_history.json')
    try:
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=4)
        logger.info(f"训练历史数据已保存至: {history_path}")
    except Exception as e:
        logger.error(f"保存训练历史数据时出错: {e}")
    # # 7.2 调用绘图函数
    # plot_path = os.path.join(experiment_output_dir, 'training_metrics.png')
    # epochs = config['training']['epochs']
    # plot_training_history(training_history, epochs, plot_path)


if __name__ == '__main__':
    main()
