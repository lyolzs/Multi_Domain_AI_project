import yaml
import argparse
import matplotlib.pyplot as plt
import logging
import os
import structlog


def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    """使用 argparse 解析命令行参数"""
    parser = argparse.ArgumentParser(description="通过命令行覆盖 YAML 配置参数")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    # 可以根据需要添加更多命令行参数来覆盖配置
    # parser.add_argument("--training.batch_size", type=int, help="训练的批量大小")
    return parser.parse_args()


def def_setup_logging(config):
    """根据配置设置全局日志记录器"""
    log_config = config['logging']
    log_file_path = log_config['file_path']

    # 确保日志文件所在的目录存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logging.basicConfig(
        level=log_config['level'].upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w',
                                encoding='utf-8'),  # 输出到文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logging.info("Logging setup complete.")


def struct_setup_logging(config, args):
    """使用 structlog 设置结构化日志记录器"""
    log_config = config['logging']
    log_level = args.log_level if args.log_level else log_config['level']

    # 1. 配置标准 logging 模块，它将作为 structlog 的输出后端
    # 我们不再需要在这里设置 format，因为 structlog 会处理格式化
    logging.basicConfig(
        level=log_level.upper(),
        stream=sys.stdout,  # 直接输出到标准输出，让 structlog 的处理器接管
        format="%(message)s",  # 格式简化，因为 structlog 会生成最终消息
    )

    # 2. 配置 structlog
    structlog.configure(
        processors=[
            # 这个处理器会添加日志级别和时间戳等上下文信息
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            # 这是核心：漂亮的、带缩进和颜色的控制台渲染器
            structlog.dev.ConsoleRenderer()
        ],
        # 将 structlog 的输出路由到标准 logging 模块
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger()
    logger.info("Logging setup complete.", log_level=log_level.upper())
# 注意：我们不再需要 FileHandler，因为 ConsoleRenderer 已经非常强大。
# 如果你仍希望有文件输出，可以配置 structlog 将 JSON 输出到文件，
# 但对于方便阅读，ConsoleRenderer 是最佳选择。
# 我们暂时简化，专注于提升可读性。


def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, epochs):
    """绘制并显示训练和测试的损失和准确率曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
