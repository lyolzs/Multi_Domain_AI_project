import matplotlib.pyplot as plt
import logging
from typing import Dict, List
import os

# 我们可以从 train 模块导入这个类型，以确保一致性
# 但为了让这个模块更独立，在这里重新定义或从一个共享的 types.py 导入更好
from src.image_classification.CNN.train import TrainingHistory

logger = logging.getLogger(__name__)


def plot_training_history(history: TrainingHistory, epochs: int, save_path: str):
    """
    根据训练历史数据绘制损失和准确率曲线，并保存到文件。

    Args:
        history (TrainingHistory): 包含训练和验证损失/准确率列表的字典。
        epochs (int): 总的训练轮数。
        save_path (str): 图像文件的保存路径。
    """
    try:
        # 确保保存路径的目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        epoch_range = range(1, epochs + 1)

        plt.style.use('seaborn-v0_8-whitegrid')  # 使用一个好看的样式
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Training and Validation Metrics', fontsize=16)

        # 绘制损失曲线
        ax1.plot(epoch_range, history['train_loss'],
                 'o-', label='Training Loss')
        ax1.plot(epoch_range, history['val_loss'],
                 's--', label='Validation Loss')
        ax1.set_title('Loss vs. Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_xticks(epoch_range)  # 确保 x 轴刻度为整数

        # 绘制准确率曲线
        ax2.plot(epoch_range, history['train_accuracy'],
                 'o-', label='Training Accuracy')
        ax2.plot(epoch_range, history['val_accuracy'],
                 's--', label='Validation Accuracy')
        ax2.set_title('Accuracy vs. Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.set_xticks(epoch_range)  # 确保 x 轴刻度为整数

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应主标题
        plt.savefig(save_path)
        plt.close(fig)  # 关闭图形，释放内存
        logger.info(f"训练历史图表已成功保存至: {save_path}")

    except Exception as e:
        logger.error(f"绘制图表时发生错误: {e}")
