from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging


def get_data_loaders(config):
    """
    根据配置创建并返回训练和测试的数据加载器
    """
    logger = logging.getLogger(__name__)  # 获取当前模块的 logger
    logger.info("Starting data loading process...")
    logger.info("Loading data...")
    logger.info("="*100)
    # 从配置中获取数据集和训练相关的配置
    dataset_config = config['dataset']
    training_config = config['training']

    logger.info("-"*100)
    logger.info(f"Using dataset: {dataset_config['name']}")
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (dataset_config['normalize_mean'],), (dataset_config['normalize_std'],))
    ])

    # 根据配置中的数据集名称加载相应的数据集
    if dataset_config['name'].lower() == 'mnist':
        dataset_class = datasets.MNIST
    elif dataset_config['name'].lower() == 'fashionmnist':
        dataset_class = datasets.FashionMNIST
    else:
        raise ValueError(f"Unsupported dataset: {dataset_config['name']}")

    # 加载数据集
    train_dataset = dataset_class(
        root=dataset_config['root'],
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = dataset_class(
        root=dataset_config['root'],
        train=False,
        download=True,
        transform=transform
    )
    logger.info(
        f"Finished: Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=training_config['batch_size'], shuffle=False)

    return train_loader, test_loader
