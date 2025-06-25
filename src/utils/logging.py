import sys
import logging
import os
import structlog


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
