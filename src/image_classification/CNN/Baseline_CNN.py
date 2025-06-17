import yaml
import os
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



# 检查并创建数据目录
if not os.path.exists('./data'):
    os.makedirs('./data')  # 如果目录不存在，则创建
print(os.getcwd())  # 输出当前工作目录
print(os.path.exists('./data'))  # 如果返回 True，说明文件夹已存在


import torch

# 检查是否支持 GPU
if torch.cuda.is_available():
    print("GPU is available!")
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("GPU is not available.")
    
    
print(torch.__version__)
print(torch.version.cuda)



import yaml

def merge_configs(base_path, override_path):
    import yaml

    with open(base_path, 'r') as f:
        base = yaml.safe_load(f)
    
    with open(override_path, 'r') as f:
        override = yaml.safe_load(f)
    
    def deep_merge(base, override):
        """
        递归深度合并字典和列表
        """
        for key, value in override.items():  # .items() 遍历字典的键值对,每个键值对以 (key, value) 的形式表示。
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                # 如果值是字典，递归合并
                deep_merge(base[key], value)
            elif isinstance(value, list) and key in base and isinstance(base[key], list):
                # 如果值是列表，合并列表（可以根据需求选择去重或直接拼接）
                # base[key].extend(value)  # 直接拼接
                base[key] = list(set(base[key] + value))  # 去重合并
            else:
                # 否则直接覆盖
                base[key] = value
    
    deep_merge(base, override)
    return base


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 加载配置
config = load_config('./参数配置/configCNN.yml')

# 打印配置查看
print(f"配置参数：\n{yaml.dump(config, default_flow_style=False, allow_unicode=True)}")


import argparse
def parse_args():
    """
    使用 argparse 解析命令行参数
    """
    parser = argparse.ArgumentParser(description="通过命令行覆盖 YAML 配置参数")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--training.batch_size", type=int, help="训练的批量大小")
    parser.add_argument("--training.learning_rate", type=float, help="学习率")
    parser.add_argument("--model.conv1_out_channels", type=int, help="第一层卷积的输出通道数")
    parser.add_argument("--model.conv2_out_channels", type=int, help="第二层卷积的输出通道数")
    # 根据需要添加更多参数
    return parser.parse_args()

def update_config_with_args(config, args):
    """
    根据命令行参数更新配置
    """
    for arg, value in vars(args).items():
        if value is not None:  # 如果命令行提供了该参数
            keys = arg.split(".")  # 将参数名按 "." 分割
            sub_config = config
            for key in keys[:-1]:  # 遍历到倒数第二层
                sub_config = sub_config.setdefault(key, {})
            sub_config[keys[-1]] = value  # 更新最终的值
    return config




# 设置随机种子以保证可重复性
torch.manual_seed(config['training']['seed'])


# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((config['dataset']['normalize_mean'],), (config['dataset']['normalize_std'],))  # 使用配置中的均值和标准差
])  # 数据处理流水线




# 加载数据集
train_dataset = datasets.MNIST(
    root=config['dataset']['root'], 
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = datasets.MNIST(
    root=config['dataset']['root'], 
    train=False, 
    download=True, 
    transform=transform
)



# 加载配置
config = load_config('./参数配置/configCNN_FashionMNIST.yml')

# 打印配置查看
print(f"配置参数：\n{yaml.dump(config, default_flow_style=False, allow_unicode=True)}")

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((config['dataset']['normalize_mean'],), (config['dataset']['normalize_std'],))  # 使用配置中的均值和标准差
])  # 数据处理流水线

train_dataset=datasets.FashionMNIST(
    root=config['dataset']['root'], 
    train=True, 
    download=True, 
    transform=transforms.ToTensor()
)
test_dataset = datasets.FashionMNIST(
    root=config['dataset']['root'], 
    train=False, 
    download=True, 
    transform=transform
)
# 查看数据集大小
print(f"训练集大小: {len(train_dataset)}")

# 查看单个样本
image, label = train_dataset[0]
print(f"图像大小: {image.shape}")  # 输出: torch.Size([1, 28, 28])
print(f"标签: {label}")  # 输出: 类别编号 (0-9)

# 可视化图像
plt.imshow(image.squeeze(), cmap='gray')  # 去掉通道维度并显示为灰度图
plt.title(f"Label: {label}")
plt.show()




# 创建数据加载器
batch_size = config['training']['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, input_channels, conv1_out_channels, conv2_out_channels, fc1_out_features, fc2_out_features):
        super(CNN, self).__init__()    # init 构造函数，用于初始化对象的属性。# 调用 nn.Module 的构造函数
        self.conv1 = nn.Conv2d(input_channels, conv1_out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(7*7*conv2_out_channels, fc1_out_features)  # 经过两次池化后图像大小为7x7
        self.dropout1 = nn.Dropout(p=config['model']['dropout']) # dropout_rate 需要在配置文件中定义
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_out_features, fc2_out_features)  # 输出类别
        self.dropout2 = nn.Dropout(p=config['model']['dropout']) 

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU device count: {torch.cuda.device_count()}")
else:
    print("No GPU available.")


model = CNN(
    input_channels=config['model']['input_channels'],
    conv1_out_channels=config['model']['conv1_out_channels'],
    conv2_out_channels=config['model']['conv2_out_channels'],
    fc1_out_features=config['model']['fc1_out_features'],
    fc2_out_features=config['model']['fc2_out_features']
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])


# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
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
        correct += pred.eq(target.view_as(pred)).sum().item()  # 计算正确预测的数量
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    train_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'\nTrain set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return train_loss, accuracy



# 测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return test_loss, accuracy


# 训练和测试循环
epochs = config['training']['epochs']
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
    test_loss, test_acc = test(model, device, test_loader, criterion)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)




import matplotlib.pyplot as plt
# 绘制训练和测试的损失和准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
temp = list(range(0, epochs ))  # 生成从1到epochs的列表，用于x轴坐标
# 在每个点上显示数据值
for x, y in zip(temp, train_losses):
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)  # 显示训练损失

for x, y in zip(temp, test_losses):
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)  # 显示测试损失
    
plt.xlabel('Epoch')
plt.xticks(range(1, epochs, 1)) 
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.xticks(range(1, epochs, 1)) 
# plt.ylim(98, 100)  # 设置y轴范围为0到100
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()


