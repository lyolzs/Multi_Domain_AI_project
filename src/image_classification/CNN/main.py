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
from .train import train_epoch, train
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
    train_loader, test_loader = get_data_loaders(config)

    # 5. 初始化模型、损失函数和优化器
    model = CNN(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    logger.info("Model, criterion, and optimizer initialized.")

    # 6. 训练和评估循环
    epochs = config['training']['epochs']
    # 直接调用 train 函数即可启动整个训练过程
    training_history = train(model, device, train_loader,
                             optimizer, criterion, epochs)

    logger.info("Training finished.")

    # # 7. 可视化结果
    # plot_metrics(training_history['train_loss'], training_history['train_accuracy'],
    #              epochs)


if __name__ == '__main__':
    main()
