import torch
import logging
import time
import datetime


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """训练模型一个 epoch"""

    logger = logging.getLogger(__name__)  # 获取当前模块的 logger
    logger.info("开始训练...")
    logger.info("="*100)

    model.to(device)

    model.train()
    train_loss = 0
    correct = 0
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

        # --- 2. 计算并打印进度 ---
        if batch_idx % 100 == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    train_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    logger.info(
        f'\nTrain set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)\n')
    return train_loss, accuracy
