import torch
import logging
import time
import datetime


# 新的顶层训练函数
def train(model, device, train_loader, optimizer, criterion, epochs):
    """
    管理整个训练流程，包含多个 epoch。
    这是从 main 文件直接调用的函数。
    """
    logger = logging.getLogger(__name__)
    logger.info("======== 开始整体训练过程 ========")

    # 1. 将模型移动到设备，此操作在整个训练流程中只需执行一次
    model.to(device)

    # 用于记录每个 epoch 的结果
    history = {'train_loss': [], 'train_accuracy': []}

    # 2. 循环指定的 epoch 次数
    for epoch in range(1, epochs + 1):

        # 3. 在每个 epoch 开始前，确保模型处于训练模式
        model.train()

        # 调用单次 epoch 的训练函数
        train_loss, train_accuracy = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch)

        # 记录当前 epoch 的结果
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        # (最佳实践) 在这里通常会调用一个 test_epoch 函数来评估模型在验证集上的表现
        # model.eval()
        # test_loss, test_accuracy = test_epoch(...)
        # history['test_loss'].append(test_loss)
        # ...

    logger.info("======== 整体训练过程结束 ========")
    return history


# 修改后的 train_epoch 函数
def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    (已重构) 仅负责训练模型一个 epoch 的逻辑。
    不再包含 model.to(device) 和 model.train()。
    """
    logger = logging.getLogger(__name__)  # 获取当前模块的 logger

    # 这些日志现在更适合放在这里，因为它描述的是单个 epoch 的行为
    logger.info(f"--- 开始训练 Epoch {epoch} ---")

    train_loss = 0
    correct = 0
    total_batches = len(train_loader)
    log_interval = max(1, total_batches // 10)  # 每 10% 打印一次

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        # 打印进度
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            logger.info(
                f'Train Epoch: {epoch} [{batch_idx + 1}/{total_batches}] | Loss: {loss.item():.6f}')

    # 计算并记录整个 epoch 的平均 loss 和准确率
    train_loss /= total_batches
    accuracy = 100. * correct / len(train_loader.dataset)
    logger.info(
        f'--- Epoch {epoch} 训练集结果: 平均 Loss: {train_loss:.4f}, 准确率: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%) ---\n')

    return train_loss, accuracy
