import torch
import torch.nn as nn
import logging
import argparse
import os
import yaml
from src.utils.set_random_seed import set_seed
from src.utils.config import load_config
from src.utils.logging import def_setup_logging
from src.image_classification.CNN.model import CNN
from src.image_classification.CNN.data_loader import get_data_loaders
# 从 train.py 中复用评估函数！
from src.image_classification.CNN.train import evaluate
from src.utils.experiment import setup_test_environment
from collections.abc import Sized


logger = logging.getLogger(__name__)


def test(weights_path: str):
    """
    独立的测试脚本，用于评估已训练好的模型在测试集上的性能。
    """
    # 2. 设置实验环境 (所有设置逻辑都被封装在这里)
    config, device = setup_test_environment(weights_path)
    # 3. 加载测试数据
    # 假设 get_data_loaders 返回 (train, val, test)
    _, _, test_loader = get_data_loaders(config)
    assert isinstance(test_loader.dataset,
                      Sized), "测试数据集必须是 map-style 并实现 __len__ 方法"
    logger.info(f"测试数据集加载完毕。共 {len(test_loader.dataset)} 个样本。")

    # 4. 初始化模型结构
    model = CNN(config).to(device)

    # 5. 加载已训练好的模型权重
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        logger.info(f"成功从 {weights_path} 加载模型权重。")
    except FileNotFoundError:
        logger.error(f"错误：找不到模型权重文件 {weights_path}。请先运行训练。")
        return

    # 6. 初始化损失函数
    criterion = nn.CrossEntropyLoss()

    # 7. 在测试集上运行评估
    logger.info("开始在测试集上进行评估...")
    test_loss, test_accuracy = evaluate(
        model, device, test_loader, criterion)

    # 8. 打印最终性能报告
    logger.info("======== 最终测试结果 ========")
    logger.info(f"  测试集平均损失: {test_loss:.4f}")
    logger.info(f"  测试集准确率:   {test_accuracy:.2f}%")
    logger.info("==============================")


if __name__ == '__main__':

    test('runs\\image_classification\\CNN_CIFAR10_2025-06-25_15-40-19\\best_cifar10_model.pth')
