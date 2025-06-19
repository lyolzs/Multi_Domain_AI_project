import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import logging  # 导入 logging 模块

# 使用相对导入，从同一目录下的其他模块导入函数和类
# 导入 setup_logging
from .utils import load_config, parse_args, plot_metrics, def_setup_logging
from .model import CNN
from .data_loader import get_data_loaders
from .train import train_epoch
from .evaluate import evaluate


def main():
    # 1. 加载配置和参数
    args = parse_args()
    config = load_config(args.config)

    # 2. 初始化日志记录器 (这是关键的第一步)
    def_setup_logging(config)
    logger = logging.getLogger(__name__)  # 获取当前模块的 logger

    logger.info(
        f"配置参数：\n{yaml.dump(config, default_flow_style=False, allow_unicode=True)}")

    # 3. 设置设备和随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['training']['seed'])
    logger.info(f"Using device: {device}")

    # 4. 加载数据
    logger.info("Loading data...")
    train_loader, test_loader = get_data_loaders(config)
    logger.info("Data loaded successfully.")

    # 5. 初始化模型、损失函数和优化器
    model = CNN(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    logger.info("Model, criterion, and optimizer initialized.")

    # 6. 训练和评估循环
    epochs = config['training']['epochs']
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, epochs + 1):
        logger.info(f"--- Starting Epoch {epoch}/{epochs} ---")
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    logger.info("Training finished.")

    # 7. 可视化结果
    plot_metrics(train_losses, test_losses,
                 train_accuracies, test_accuracies, epochs)


if __name__ == '__main__':
    main()
