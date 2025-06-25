from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import logging
from datasets import load_dataset  # 从 Hugging Face 数据集库加载数据集
import datasets as hf_datasets
from typing import Optional, Tuple

logger = logging.getLogger(__name__)  # 获取当前模块的 logger

DatasetSplits = Tuple[hf_datasets.Dataset,
                      Optional[hf_datasets.Dataset], hf_datasets.Dataset]


class HFDatasetWrapper(Dataset):
    """Hugging Face aasetset 的包装器，使其与 torchvision transform 兼容并返回 (image, label) 元组。
    这个类自动检测图像和标签列的名称，支持不同数据集的灵活性。
    Args:
        hf_dataset (datasets.Dataset): Hugging Face 数据集对象
        transform (callable, optional): 应用于图像的转换函数
    """

    def __init__(self, hf_dataset: hf_datasets.Dataset, transform: Optional[transforms.Compose] = None):
        self.hf_dataset: hf_datasets.Dataset = hf_dataset
        self.transform: Optional[transforms.Compose] = transform
        # 自动检测图像和标签列的名称
        self.image_key: str = 'image' if 'image' in hf_dataset.features else 'img'
        self.label_key: str = 'label'

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """
        根据索引从数据集中检索一个样本。

        Args:
            idx (int): 要检索的样本的索引。

        Returns:
            tuple: 一个包含 (image, label) 的元组。如果定义了 transform,image 会被转换。
        """
        item: dict = self.hf_dataset[idx]
        image: list = item[self.image_key]
        label: list = item[self.label_key]

        if self.transform:
            image: list = self.transform(image)

        return image, label


def _load_data(config: dict) -> hf_datasets.DatasetDict:
    """
    根据配置文件加载对应数据集并返回完整的 DatasetDict 对象。

    Args:
        config (dict): 包含数据集和训练相关配置的字典。
                       需要 `config['dataset']['name']` 和 `config['dataset']['root']`。

    Returns:
        hf_datasets.DatasetDict: 从 Hugging Face Hub 加载的完整数据集对象。

    Raises:
        TypeError: 如果从 Hugging Face Hub 加载的对象不是 `DatasetDict` 类型。
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
    """
    根据配置创建训练和测试所需的数据转换流程。

    支持为训练集启用或禁用数据增强。

    Args:
        config (dict): 包含数据集配置的字典。
                       需要 `config['dataset']['image_size']`,
                       `config['dataset']['normalize_mean']`,
                       `config['dataset']['normalize_std']`,
                       以及可选的 `config['dataset']['data_augmentation']`。

    Returns:
        tuple[transforms.Compose, transforms.Compose]: 一个包含两个元素的元组，
            分别是训练集和测试集的数据转换流程。
    """
    dataset_config = config['dataset']
    image_size = dataset_config['image_size']

    mean = dataset_config['normalize_mean']
    std = dataset_config['normalize_std']
    if not isinstance(mean, (list, tuple)):
        mean = [mean]
    if not isinstance(std, (list, tuple)):
        std = [std]
    # 为测试集和验证集定义确定性的转换
    test_transform: transforms.Compose = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # 为训练集定义带数据增强的转换
    # 通过配置决定是否启用数据增强
    if dataset_config.get('data_augmentation', True):
        logger.info("Using data augmentation for the training set.")
        train_transform: transforms.Compose = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        logger.info(
            "Data augmentation disabled. Using basic resize for the training set.")
        train_transform: transforms.Compose = test_transform
    return train_transform, test_transform


def _prepare_dataset_splits(full_dataset: hf_datasets.DatasetDict, config: dict) -> DatasetSplits:
    """
    从完整数据集中准备训练、验证和测试的分割，并处理各种数据集分割情况。

    处理逻辑:
    1. 创建数据集副本以进行安全操作。
    2. 优先使用 'test' 分割作为测试集。若无，则使用 'validation' 分割作为测试集。
    3. 使用 'train' 分割作为训练集。
    4. 若数据集中有独立的 'validation' 分割，则使用它。否则，根据配置从训练集中切分出验证集。

    Args:
        full_dataset (hf_datasets.DatasetDict): 原始的、包含所有分割的数据集。
        config (dict): 包含数据集和训练配置的字典。
                       若需切分验证集，则要提供 `config['dataset']['validation_split_ratio']`
                       和 `config['training']['seed']`。

    Returns:
        Tuple[hf_datasets.Dataset, Optional[hf_datasets.Dataset], hf_datasets.Dataset]: 
        一个元组，包含处理好的训练集、验证集 (可能为 None) 和测试集。

    Raises:
        KeyError: 如果数据集中找不到 'train' 分割，或者找不到可用于测试的 'test' 或 'validation' 分割。
    """
    dataset_config: dict = config['dataset']
    training_config: dict = config['training']

    dataset_splits: hf_datasets.DatasetDict = hf_datasets.DatasetDict(
        full_dataset)
    # 1. 确定测试集
    if 'test' in full_dataset:
        test_hf: hf_datasets.Dataset = dataset_splits.pop('test')
        logger.info("Using 'test' split for final testing.")
    elif 'validation' in full_dataset:
        test_hf = dataset_splits.pop('validation')
        logger.warning(
            "No 'test' split found. Using 'validation' split for final testing.")
    else:
        available_splits = list(full_dataset.keys())
        logger.error(
            f"在数据集中找不到 'test' 或 'validation' 分割，可用分割: {available_splits}")
        raise KeyError(
            f"Could not find a 'test' or 'validation' split in dataset. Available splits: {available_splits}")

    # 2. 准备训练集和验证集
    train_hf: hf_datasets.Dataset = dataset_splits['train']
    val_hf: Optional[hf_datasets.Dataset] = None
    if 'validation' in dataset_splits:
        logger.info("Using dedicated 'validation' split from the dataset.")
        val_hf = dataset_splits['validation']
    else:
        validation_split_ratio = dataset_config.get(
            'validation_split_ratio', 0.2)
        if validation_split_ratio > 0:
            logger.info(
                f"Creating validation set by splitting 'train' set with ratio: {validation_split_ratio}.")
            seed = training_config.get('seed', 42)
            train_val_split: hf_datasets.DatasetDict = train_hf.train_test_split(
                test_size=validation_split_ratio, seed=seed)
            train_hf: hf_datasets.Dataset = train_val_split['train']
            val_hf = train_val_split['test']
        else:
            logger.info(
                "No validation set will be created as 'validation_split_ratio' is 0.")

    return train_hf, val_hf, test_hf


def get_data_loaders(config: dict) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    数据加载的主协调函数，执行从加载数据到创建 DataLoader 的完整流程。

    流程步骤:
    1. 调用 `_load_data` 加载原始数据集。
    2. 调用 `_create_transforms` 创建数据转换。
    3. 调用 `_prepare_dataset_splits` 准备数据集分割。
    4. 将分割后的数据集包装成 `HFDatasetWrapper`。
    5. 创建并返回训练、验证和测试的 `DataLoader`。

    Args:
        config (dict): 完整的项目配置字典，包含了所有子模块所需的配置。

    Returns:
        Tuple    [DataLoader, Optional[DataLoader], DataLoader]:
            一个元组，包含训练、验证 (可能为 None) 和测试的数据加载器。
    """
    # 1: 加载完整数据集
    full_dataset: hf_datasets.DatasetDict = _load_data(config)
    # 2: 创建训练和测试转换
    train_transform, test_transform = _create_transforms(config)
    # 3: 准备数据集分割
    train_hf, val_hf, test_hf = _prepare_dataset_splits(full_dataset, config)
    # 4: 创建 PyTorch 数据集和数据加载器
    train_dataset = HFDatasetWrapper(train_hf, transform=train_transform)
    test_dataset = HFDatasetWrapper(test_hf, transform=test_transform)

    training_config = config['training']
    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    test_loader: DataLoader = DataLoader(
        test_dataset, batch_size=training_config['batch_size'], shuffle=False)
    val_loader: Optional[DataLoader] = None

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
    from src.utils.logging import def_setup_logging
    with open('configs/config.yml', 'r') as f:
        config: dict = yaml.safe_load(f)
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
