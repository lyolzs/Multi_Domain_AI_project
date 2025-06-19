import torch.nn as nn


class CNN(nn.Module):
    """
    一个简单的卷积神经网络模型
    """

    def __init__(self, config):
        super(CNN, self).__init__()
        model_config = config['model']
        self.conv1 = nn.Conv2d(
            model_config['input_channels'], model_config['conv1_out_channels'], kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            model_config['conv1_out_channels'], model_config['conv2_out_channels'], kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(
            7*7*model_config['conv2_out_channels'], model_config['fc1_out_features'])
        self.dropout1 = nn.Dropout(p=model_config['dropout'])
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(
            model_config['fc1_out_features'], model_config['fc2_out_features'])

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(x.size(0), -1)  # 展平

        x = self.relu3(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
