import yaml
import argparse


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
