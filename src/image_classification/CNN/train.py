import torch
import torch.nn as nn
import logging
import time
import os
import sys
import datetime
from typing import Tuple, Optional, Dict, List, TypedDict
from collections.abc import Sized
from .hooks import forward_hook, backward_grad_hook  # 从新文件中导入
from .model import CNN

logger = logging.getLogger(__name__)
# 新的顶层训练函数


class TrainingHistory(TypedDict):
    """描述完整训练过程的历史记录。"""
    train_loss: List[float]
    train_accuracy: List[float]
    val_loss: List[float]
    val_accuracy: List[float]


def train_epoch(model: nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, epoch: int) -> Tuple[float, float]:
    """仅负责训练模型一个 epoch。"""
    model.train()  # 设置为训练模式
    train_loss = 0.0
    correct = 0
    total = 0
    # 这些日志现在更适合放在这里，因为它描述的是单个 epoch 的行为
    logger.info(f"--- 开始训练 Epoch {epoch} ---")
    assert isinstance(train_loader.dataset,
                      Sized), "训练数据集必须是 map-style 并实现 __len__ 方法"

    for batch_idx, (data, labels) in enumerate(train_loader, start=1):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # pred 形状为 [64, 1]
        # 使用 .view_as() 让 label 的形状也变为 [64, 1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_loss = train_loss / len(train_loader)
    total_samples = len(train_loader.dataset)
    accuracy = 100. * correct / total_samples
    logger.info(
        f'--- epoch训练集: {epoch}\t平均损失: {avg_loss:.4f}\t准确率: {correct}/{total} ({accuracy:.2f}%)')
    return avg_loss, accuracy

# 函数二：单次评估循环


def evaluate(model: nn.Module, device: torch.device, val_loader: torch.utils.data.DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()  # 2. 设置为评估模式
    test_loss = 0.0
    correct = 0
    assert isinstance(val_loader.dataset,
                      Sized), "训练数据集必须是 map-style 并实现 __len__ 方法"
    with torch.no_grad():  # 3. 关闭梯度计算
        for eval_idx, (data, labels) in enumerate(val_loader):
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            loss = criterion(output, labels)

            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # pred 形状为 [64, 1]
            # 使用 .view_as() 让 label 的形状也变为 [64, 1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_loss = test_loss / len(val_loader)
    total_samples = len(val_loader.dataset)
    accuracy = 100. * correct / total_samples
    logger.info(
        f'------ 验证集: \t平均损失: {avg_loss:.4f}\t准确率: {correct}/{total_samples} ({accuracy:.2f}%)')
    return avg_loss, accuracy


def train(*, model: nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, config: dict, output_dir: str) -> TrainingHistory:
    """
    管理整个训练流程，包含多个 epoch。
    这是从 main 文件直接调用的函数。
    """
    logger.info("======== 开始整体训练过程 ========")
    epochs = config['training']['epochs']
    model_filename = config['training'].get(
        'model_filename', 'best_model.pth')  # 提供默认文件名
    model_save_path = os.path.join(output_dir, model_filename)
    # 确保输出目录存在
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    model.to(device)
    # [新增] 记录训练和验证历史
    history: TrainingHistory = {'train_loss': [], 'train_accuracy': [],
                                'val_loss': [], 'val_accuracy': []}
    best_val_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # 1. 训练阶段
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch)

        # 2. 验证阶段
        val_loss, val_acc = evaluate(
            model, device, val_loader, criterion)

        end_time = time.time()
        epoch_duration = str(datetime.timedelta(
            seconds=int(end_time - start_time)))
        logger.info(f"--- Epoch {epoch} 结束, 耗时: {epoch_duration} ---\n")

        # 3. 记录历史数据
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # 4. 检查并保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)
            logger.info(
                f"发现新的最佳模型！验证准确率: {best_val_accuracy:.2f}%. 模型已保存至 {model_save_path}")

    logger.info(
        f"======== 整体训练过程结束, 最佳验证准确率为: {best_val_accuracy:.2f}% ========")
    return history
