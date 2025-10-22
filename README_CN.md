# EGA-Ploc: 用于多标签蛋白质亚细胞定位的高效全局-局部注意力模型

[![许可证](https://img.shields.io/badge/许可证-学术自由许可证v3.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-red.svg)](https://pytorch.org/)
[![论文](https://img.shields.io/badge/论文-IEEE%20JBHI-green.svg)](https://ieeexplore.ieee.org/document/11175487)

> **语言**: [English](README.md) | [中文](README_CN.md)

---

## 📖 概述

**EGA-Ploc** 是一个先进的深度学习框架，用于从免疫组织化学（IHC）图像中预测多标签蛋白质亚细胞定位。本仓库包含我们论文的官方实现：

> **[EGA-Ploc: 基于免疫组织化学图像的多标签蛋白质亚细胞定位预测的高效全局-局部注意力模型](https://ieeexplore.ieee.org/document/11175487)**  
> *已被 IEEE Journal of Biomedical and Health Informatics 接收，2025年9月*

### 🎯 主要特性

- **🔬 新颖的线性注意力机制**: 高效地从IHC图像中捕获判别性表示
- **🏗️ 分层多尺度架构**: 保留细粒度亚细胞模式和全局空间上下文
- **⚖️ 增强的多标签目标**: 通过联合优化抵消数据集不平衡
- **🚀 高性能**: 优于现有的基于补丁和依赖下采样的方法

## 🚀 快速开始

### 在线演示

通过我们的 Hugging Face Space 即时体验 EGA-Ploc：  
👉 **[实时演示](https://huggingface.co/spaces/austinx25/EGA-Ploc)**

### 本地安装

1. **克隆仓库**
   ```bash
   git clone https://github.com/gd-hxy/EGA-Ploc.git
   cd EGA-Ploc
   ```

2. **设置环境**
   ```bash
   conda env create -f environment.yaml
   conda activate Vislocas
   ```

3. **下载数据集**
   - **Vislocas 数据集**: [Zenodo](https://doi.org/10.5281/zenodo.10632698)
   - **HPA18 数据集**: [GraphLoc](http://www.csbio.sjtu.edu.cn/bioinf/GraphLoc)

4. **下载预训练模型** (可选)
   - **Vislocas 模型**: [下载链接](https://jxstnueducn-my.sharepoint.com/:f:/g/personal/wanboyang_jxstnu_edu_cn/EpEDB3GcXMZFvRz9lQaBHswBYTEWUDF6ThPBHWqEPB-eUQ?e=jsSoY0)
   - **HPA18 模型**: 同上仓库

## 🏗️ 模型架构

EGA-Ploc 集成了三个关键组件：

1. **高效全局-局部注意力**: 新颖的线性注意力机制，计算效率高
2. **多尺度特征融合**: 分层架构捕获局部和全局模式
3. **平衡多标签学习**: 增强的目标函数解决类别不平衡

### 模型变体

- **ETP_cls_l1/l2/l3**: 具有不同骨干复杂度的基础模型变体
- **ETP_cls_cl0/cl1/cl2/cl3**: 级联骨干变体
- **特征融合模型**: 多尺度特征集成变体 (featureAdd234, featureAdd324 等)

## 📊 性能

EGA-Ploc 在 Vislocas 和 HPA18 数据集上都实现了最先进的性能，在多标签蛋白质亚细胞定位任务中表现出卓越的准确性。

## 📁 项目结构

```
EGA-Ploc/
├── assets/                 # 演示测试的样本图像
├── data/                   # 数据集标注文件 (.csv)
├── datasets/               # 数据加载模块
│   ├── ihc.py             # Vislocas 数据集加载器
│   ├── HPA18.py           # HPA18 数据集加载器
│   ├── build.py           # 数据集构建器
│   └── loader.py          # 数据加载工具
├── models/                 # 核心模型实现
│   └── ETPLoc/            # EGA-Ploc 模型架构
│       ├── backbone.py    # 骨干网络
│       ├── cls.py         # 分类头
│       ├── nn/            # 神经网络模块
│       └── utils/         # 模型工具
├── tools/                  # 训练和测试脚本
│   ├── test.py            # 完整数据集评估
│   ├── test_demo.py       # 单图像测试
│   └── train.py           # 模型训练
├── utils/                  # 工具函数
│   ├── args.py            # 命令行参数
│   ├── checkpoint.py      # 模型检查点
│   ├── eval_metrics.py    # 性能评估
│   └── optimizer.py       # 优化算法
└── results/               # 模型和预测的输出目录
```

## 🛠️ 使用

### 完整数据集测试

```bash
# 在 Vislocas 数据集上测试
python tools/test.py --dataset IHC

# 在 HPA18 数据集上测试  
python tools/test.py --dataset HPA18
```

### 单图像测试

```bash
# 测试单张图像
python tools/test_demo.py --dataset IHC --single_image_path ./assets/Cytopl;Mito/55449_A_1_2.jpg
```

### 训练

EGA-Ploc 支持多GPU分布式训练，以实现高效的模型训练。训练过程包括以下关键组件：

#### 训练设置

```bash
# 使用8个GPU启动分布式训练
bash train.sh

# 或直接使用Python运行
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py
```

#### 训练配置

训练过程通过 `utils/config_defaults.py` 配置，包含以下关键参数：

- **模型架构**: 多个 EGA-Ploc 变体，包括 `AIP_discount_fa_4_cl1_3000_wd-005_mlce`
- **训练轮数**: 120轮，使用预热余弦调度器
- **学习率**: 5e-5，使用 AdamW 优化器
- **批大小**: 1（由于高分辨率 IHC 图像）
- **损失函数**: 多标签平衡交叉熵，处理类别不平衡
- **正则化**: L1 和 L2 正则化，权重可配置
- **混合精度**: 启用以提高内存效率

#### 训练过程

1. **数据加载**: 
   - 从 CSV 标注文件加载训练和验证数据集
   - 支持 Vislocas 和 HPA18 数据集
   - 对训练集应用数据增强

2. **模型初始化**:
   - 构建 EGA-Ploc 分类器模型
   - 支持分布式数据并行（DDP）训练
   - 将批归一化转换为同步批归一化以支持多GPU训练

3. **优化**:
   - **优化器**: AdamW，权重衰减可配置（0.05、0.01、0.005 或 0）
   - **调度器**: 预热余弦退火调度器
   - **梯度缩放**: 自动混合精度（AMP）以提高内存效率

4. **损失函数**:
   - **多标签平衡交叉熵**: 处理蛋白质定位中的类别不平衡
   - **带Logits的BCE**: 标准二元交叉熵选项
   - **多标签分类交叉熵**: 替代损失函数

5. **训练循环**:
   - 遍历训练数据，定期验证
   - 基于验证损失保存最佳模型检查点
   - 支持中断训练恢复检查点
   - 记录训练进度和指标

6. **评估**:
   - 每5步定期验证
   - 全面的评估指标，包括精确率、召回率、F1分数
   - 多GPU同步评估

#### 支持的数据集

- **Vislocas**: 5个定位类别（细胞质、内质网、线粒体、细胞核、质膜）
- **HPA18**: 7个定位类别（细胞质、高尔基体、线粒体、细胞核、内质网、质膜、囊泡）

#### 模型检查点

训练自动保存：
- **最新模型**: 用于恢复中断的训练
- **最佳模型**: 基于验证性能
- **训练日志**: TensorBoard 兼容的日志，用于监控

开始训练前，请确保已下载所需数据集并在配置文件中配置了适当的数据路径。

## 📋 要求

- **平台**: Ubuntu 20.04+（支持 Windows/Linux）
- **GPU**: NVIDIA GPU，CUDA 11.3+
- **Python**: 3.8.15
- **PyTorch**: 1.11.0
- **依赖项**: 参见 `environment.yaml` 获取完整列表

## 📄 许可证

### 学术使用
本项目根据**学术自由许可证 v3.0** 发布，用于非商业研究和教育目的。您可以：

- ✅ 为学术研究使用、复制和修改
- ✅ 为教育目的共享和分发
- ✅ 为非商业应用程序在此工作基础上构建

### 商业使用
对于商业许可，请联系作者讨论条款和条件。商业使用需要明确许可，并可能需要支付许可费。

## 🤝 引用

如果您在研究中使用了 EGA-Ploc，请引用我们的论文：

```bibtex
@article{wan2025ega,
  title={EGA-Ploc: An Efficient Global-Local Attention Model for Multi-label Protein Subcellular Localization Prediction on the Immunohistochemistry Images},
  author={Wan, Boyang and Huang, Xiaoyang and Qiao, Yang and Peng, Jiajie and Yang, Fan},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}
```

## 📞 联系方式

如有问题、问题或合作机会：

- **邮箱**: `wanboyangjerry@163.com`
- **问题**: [GitHub Issues](https://github.com/gd-hxy/EGA-Ploc/issues)

## 🙏 致谢

我们感谢贡献者和研究社区在开发 EGA-Ploc 过程中提供的宝贵反馈和支持。

---

*EGA-Ploc: 通过高效的深度学习架构推进蛋白质定位预测。*
