import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class CNN(nn.Module):
    """
    一个简单的卷积神经网络模型
    """

    def __init__(self, config: dict):
        super(CNN, self).__init__()
        model_config: dict = config['model']
        dataset_config: dict = config['dataset']

        # 1. 定义特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(
                model_config['input_channels'], model_config['conv1_out_channels'], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                model_config['conv1_out_channels'], model_config['conv2_out_channels'], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 2. 动态计算全连接层的输入尺寸
        # 创建一个与输入图像尺寸相同的虚拟张量
        dummy_input_size: list = [1, model_config['input_channels']
                                  ] + dataset_config['image_size']
        # 解包[1, 3, 224, 224]为 (1, 3, 224, 224)
        dummy_input = torch.randn(*dummy_input_size)
        # 通过一次“伪正向传播”来获取特征图的尺寸
        dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
        logger.info(f"动态计算出的全连接层输入尺寸为: {flattened_size}")
        # 3. 定义分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, model_config['fc1_out_features']),
            nn.ReLU(inplace=True),
            nn.Dropout(p=model_config['dropout']),
            nn.Linear(model_config['fc1_out_features'],
                      model_config['fc2_out_features'])
        )

    def forward(self, x):

        # 特征提取
        x = self.features(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 分类
        x = self.classifier(x)
        return x
