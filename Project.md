# 算法学习项目库

---

## 项目结构

```text
multi_domain_ai_project/
├── configs/                          # 顶级配置文件目录
│   ├── image_classification/         # 图像分类任务的配置
│   │   ├── cnn_mnist_config.yml
│   │   └── resnet_cifar10_config.yml
│   ├── natural_language_processing/  # NLP 任务的配置
│   │   ├── transformer_sentiment_config.yml
│   │   └── bert_ner_config.yml
│   └── object_detection/             # 目标检测任务的配置
│       ├── yolo_coco_config.yml
│       └── ssd_pascal_config.yml
│
├── src/                              # 顶级源代码目录 (或者可以命名为 "pipelines", "domains", "applications")
│   ├── image_classification/         # 图像分类子项目
│   │   ├── __init__.py
│   │   ├── data_loader.py            # 图像分类特有的数据加载
│   │   ├── models/                   # 存放 CNN 等图像分类模型
│   │   │   ├── __init__.py
│   │   │   ├── cnn_architectures.py
│   │   │   └── resnet_architectures.py
│   │   ├── trainer.py                # 图像分类特有的训练逻辑
│   │   ├── evaluate.py
│   │   └── main.py                   # 图像分类任务的入口脚本
│   │
│   ├── natural_language_processing/  # NLP 子项目
│   │   ├── __init__.py
│   │   ├── data_loader.py            # NLP 特有的数据加载 (分词、编码等)
│   │   ├── models/                   # 存放 Transformer 等 NLP 模型
│   │   │   ├── __init__.py
│   │   │   └── transformer_architectures.py
│   │   ├── trainer.py                # NLP 特有的训练逻辑
│   │   ├── evaluate.py
│   │   └── main.py                   # NLP 任务的入口脚本
│   │
│   ├── object_detection/             # 目标检测子项目
│   │   ├── __init__.py
│   │   ├── data_loader.py            # 目标检测特有的数据加载 (边界框、锚点等)
│   │   ├── models/                   # 存放 YOLO, SSD 等目标检测模型
│   │   │   ├── __init__.py
│   │   │   └── yolo_architectures.py
│   │   ├── trainer.py                # 目标检测特有的训练逻辑
│   │   ├── evaluate.py
│   │   └── main.py                   # 目标检测任务的入口脚本
│   │
│   └── common_utils/                 # **真正通用的**工具函数 (非常少)
│       ├── __init__.py
│       ├── logger_setup.py           # 例如，全局日志配置
│       └── generic_config_loader.py  # 一个非常通用的配置加载器 (如果可能)
│
├── data/                             # 顶级数据目录，内部按领域划分
│   ├── image_classification_data/
│   ├── nlp_data/
│   └── object_detection_data/
│
├── notebooks/                        # Jupyter Notebooks，也建议按领域划分
│   ├── image_classification_eda.ipynb
│   └── nlp_experiments.ipynb
│
├── runs/                             # 实验输出，也建议按领域划分
│   ├── image_classification/
│   │   └── experiment_timestamp/
│   ├── natural_language_processing/
│   │   └── experiment_timestamp/
│   └── object_detection/
│       └── experiment_timestamp/
│
├── requirements.txt                  # 项目总的依赖 (如果依赖冲突严重，可能需要为每个子项目管理)
├── Makefile                          # 顶级 Makefile，用于协调不同子项目的任务
└── README.md                         # 项目总述，说明如何运行各个子项目
```

## 任务列表

- [x] **构建项目目录结构**
  - [ ] *图像识别 CNN 目录结构*
- [ ] **CNN 图像识别算法入库**
- [ ] **utils 辅助代码库构建**
  - [ ] *Log 日志记录*
  - [ ]
