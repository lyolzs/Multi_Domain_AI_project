import os
import time
import shutil
import logging
import argparse
from typing import Tuple

import torch
import yaml

from .logging import def_setup_logging
from .config import load_config
from .set_random_seed import set_seed

logger = logging.getLogger(__name__)


def setup_train_environment(args: argparse.Namespace) -> Tuple[dict, torch.device, str]:
    """
    设置整个实验环境，包括配置、目录、日志、设备和随机种子。

    Args:
        args: 从命令行解析的参数。

    Returns:
        一个元组，包含:
        - config (dict): 加载并处理后的配置字典。
        - device (torch.device): 计算设备 (CPU 或 GPU)。
        - experiment_output_dir (str): 本次实验唯一的输出目录路径。
    """
    # 1. 加载配置和设置种子
    config = load_config(args.config)
    set_seed(config['training']['seed'])

    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 创建唯一的实验输出目录
    experiment_name = config.get('experiment', {}).get('name', 'experiment')
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    unique_folder_name = f"{experiment_name}_{timestamp}"
    base_output_dir = config.get(
        'experiment', {}).get('output_dir', 'runs')
    experiment_output_dir = os.path.join(base_output_dir, unique_folder_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    # 4. 设置日志记录器，并将配置快照保存到实验目录
    log_filename = config['logging'].get('log_filename', 'run.log')
    config['logging']['file_path'] = os.path.join(
        experiment_output_dir, log_filename)
    def_setup_logging(config)  # 初始化日志
    shutil.copy(args.config, os.path.join(experiment_output_dir, 'config.yml'))
    logger.info(
        f"================== 开始实验({config['experiment']['name']}) ==================\n"
        f"配置参数：\n{yaml.dump(config, default_flow_style=False, allow_unicode=True)}"
        f"\n实验输出目录: {experiment_output_dir}"
        f"\n日志文件: {config['logging']['file_path']}"
        f"\nDevice: {device}")

    return config, device, experiment_output_dir


def setup_test_environment(weights_path: str) -> Tuple[dict, torch.device]:
    """
    为测试脚本设置环境，它会加载一个已存在的实验配置。

    Args:
        weights_path: 已训练好的模型权重文件路径。

    Returns:
        一个元组，包含:
        - config (dict): 从实验目录加载的配置字典。
        - device (torch.device): 计算设备 (CPU 或 GPU)。
    """
    # 1. 从权重路径推断出实验目录和配置文件路径
    experiment_dir = os.path.dirname(weights_path)
    config_path = os.path.join(experiment_dir, 'config.yml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"在实验目录中找不到配置文件: {config_path}")

    # 2. 加载配置并设置设备
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. [关键] 配置日志以写入 test.log
    config['logging']['file_path'] = os.path.join(experiment_dir, "test.log")
    def_setup_logging(config)

    logger.info("=" * 80)
    logger.info(f"开始测试: '{config['experiment']['name']}'")
    logger.info(
        f"配置参数: {yaml.dump(config, default_flow_style=False, allow_unicode=True)}")
    logger.info(f"从目录加载配置: {experiment_dir}")
    logger.info(f"使用设备: {device}")
    logger.info("=" * 80)

    return config, device
