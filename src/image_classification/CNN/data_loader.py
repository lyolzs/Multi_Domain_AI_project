from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import logging
from datasets import load_dataset  # 从 Hugging Face 数据集库加载数据集
import datasets as hf_datasets
from .utils import def_setup_logging

logger = logging.getLogger(__name__)  # 获取当前模块的 logger


class HFDatasetWrapper(Dataset):
    """Hugging Face aasetset 的包装器，使其与 torchvision transform 兼容并返回 (image, label) 元组。
    这个类自动检测图像和标签列的名称，支持不同数据集的灵活性。
    Args:
        hf_dataset (datasets.Dataset): Hugging Face 数据集对象
        transform (callable, optional): 应用于图像的转换函数
    """

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        # 自动检测图像和标签列的名称
        self.image_key = 'image' if 'image' in hf_dataset.features else 'img'
        self.label_key = 'label'

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item[self.image_key]
        label = item[self.label_key]

        if self.transform:
            image = self.transform(image)

        return image, label


def _load_data(config: dict) -> hf_datasets.DatasetDict:
    """根据配置文件加载对应数据集并返回完整的数据集对象
    Args:
        config (dict): 包含数据集和训练相关配置的字典
    Returns:
        full_dataset (datasets.Dataset): 完整的数据集对象
    """
    # 从配置中获取数据集和训练相关的配置
    dataset_config = config['dataset']
    dataset_name = dataset_config['name'].lower()

    # Hugging Face Hub 使用 'fashion_mnist' 而不是 'fashionmnist'
    if dataset_name == 'fashionmnist':
        dataset_name = 'fashion_mnist'

    logger.info(f"Using dataset: {dataset_name} from Hugging Face Hub")
    # 从 Hugging Face Hub 加载数据集
    # 'root' 配置可用作缓存目录
    full_dataset = load_dataset(
        dataset_name, cache_dir=dataset_config['root'])
    # 验证加载的数据集是否为预期的 DatasetDict 类型
    if not isinstance(full_dataset, hf_datasets.DatasetDict):
        raise TypeError(
            f"期望 load_dataset 返回一个 DatasetDict,但得到了 {type(full_dataset)}。"
            f"请检查 Hugging Face Hub 上的数据集 '{dataset_name}' 是否包含标准的分割（如 'train', 'test'）。"
        )
    # 检查数据集是否成功加载
    logger.info(
        f"加载完成-Dataset: '{full_dataset}'\n ")
    return full_dataset


def _create_transforms(config: dict) -> tuple[transforms.Compose, transforms.Compose]:
    dataset_config = config['dataset']
    image_size = dataset_config['image_size']

    mean = dataset_config['normalize_mean']
    std = dataset_config['normalize_std']
    if not isinstance(mean, (list, tuple)):
        mean = [mean]
    if not isinstance(std, (list, tuple)):
        std = [std]
    # 为测试集和验证集定义确定性的转换
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # 为训练集定义带数据增强的转换
    # 通过配置决定是否启用数据增强
    if dataset_config.get('data_augmentation', True):
        logger.info("Using data augmentation for the training set.")
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        logger.info(
            "Data augmentation disabled. Using basic resize for the training set.")
        train_transform = test_transform
    return train_transform, test_transform


def _prepare_dataset_splits(full_dataset: hf_datasets.DatasetDict, config: dict):
    """从完整数据集中准备训练、验证和测试的分割。"""
    dataset_config = config['dataset']
    training_config = config['training']

    # 1. 确定测试集
    if 'test' in full_dataset:
        test_split_name = 'test'
    elif 'validation' in full_dataset:
        test_split_name = 'validation'
    else:
        available_splits = list(full_dataset.keys())
        logger.error(
            f"在数据集中找不到 'test' 或 'validation' 分割，可用分割: {available_splits}")
        raise KeyError(
            f"Could not find a 'test' or 'validation' split in dataset. Available splits: {available_splits}")
    test_hf = full_dataset[test_split_name]

    # 2. 准备训练集和验证集
    train_hf = full_dataset['train']
    val_hf = None
    if 'validation' in full_dataset:
        logger.info("Using dedicated 'validation' split from the dataset.")
        val_hf = full_dataset['validation']
    else:
        validation_split_ratio = dataset_config.get(
            'validation_split_ratio', 0.1)
        if validation_split_ratio > 0:
            logger.info(
                f"Creating validation set by splitting 'train' set with ratio: {validation_split_ratio}.")
            seed = training_config.get('seed', 42)
            train_val_split = train_hf.train_test_split(
                test_size=validation_split_ratio, seed=seed)
            train_hf = train_val_split['train']
            val_hf = train_val_split['test']
        else:
            logger.info(
                "No validation set will be created as 'validation_split_ratio' is 0.")

    return train_hf, val_hf, test_hf


def get_data_loaders(config: dict):
    """
    根据配置创建并返回训练验证和测试的数据加载器
    """
    # 加载完整数据集
    full_dataset = _load_data(config)
    # 创建训练和测试转换
    train_transform, test_transform = _create_transforms(config)
    # 步骤 3: 准备数据集分割
    train_hf, val_hf, test_hf = _prepare_dataset_splits(full_dataset, config)
    # 步骤 4: 创建 PyTorch 数据集和数据加载器
    train_dataset = HFDatasetWrapper(train_hf, transform=train_transform)
    test_dataset = HFDatasetWrapper(test_hf, transform=test_transform)

    training_config = config['training']
    train_loader = DataLoader(
        train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=training_config['batch_size'], shuffle=False)
    val_loader = None

    if val_hf:
        val_dataset = HFDatasetWrapper(
            val_hf, transform=test_transform)  # 验证集不使用数据增强
        val_loader = DataLoader(
            val_dataset, batch_size=training_config['batch_size'], shuffle=False)
        logger.info(
            f"Finished: Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    else:
        logger.info(
            f"Finished: Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载器
    import yaml
    with open('configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    def_setup_logging(config)

    logger.info(
        f"Loading configuration... \n {yaml.dump(config, allow_unicode=True)}")

    train_loader, val_loader, test_loader = get_data_loaders(config)
    logger.info(
        f"Train loader size: {len(train_loader)}, Test loader size: {len(test_loader)}")
    for images, labels in train_loader:
        logger.info(
            f"Batch size: {images.size(0)}, Image shape: {images.shape}, Labels: {labels}")
        break  # 只打印第一个批次的信息
