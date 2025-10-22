# EGA-Ploc: Efficient Global-Local Attention Model for Multi-label Protein Subcellular Localization

[![License](https://img.shields.io/badge/License-Academic%20Free%20License%20v3.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-red.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20JBHI-green.svg)](https://ieeexplore.ieee.org/document/11175487)

> **Language**: [English](README.md) | [中文](README_CN.md)

---

## 📖 Overview

**EGA-Ploc** is an advanced deep learning framework for multi-label protein subcellular localization prediction from immunohistochemistry (IHC) images. This repository contains the official implementation of our paper:

> **[EGA-Ploc: An Efficient Global-Local Attention Model for Multi-label Protein Subcellular Localization Prediction on the Immunohistochemistry Images](https://ieeexplore.ieee.org/document/11175487)**  
> *Accepted by IEEE Journal of Biomedical and Health Informatics, September 2025*

### 🎯 Key Features

- **🔬 Novel Linear Attention Mechanism**: Efficiently captures discriminative representations from IHC images
- **🏗️ Hierarchical Multi-scale Architecture**: Preserves both fine-grained subcellular patterns and global spatial context
- **⚖️ Enhanced Multi-label Objective**: Counteracts dataset imbalance through joint optimization
- **🚀 High Performance**: Outperforms existing patch-based and downsampling-reliant approaches

## 🚀 Quick Start

### Online Demo

Experience EGA-Ploc instantly through our Hugging Face Space:  
👉 **[Live Demo](https://huggingface.co/spaces/austinx25/EGA-Ploc)**

### Local Installation

1. **Clone the repository**
   
   ```bash
   git clone https://github.com/gd-hxy/EGA-Ploc.git
   cd EGA-Ploc
   ```

2. **Set up environment**
   
   ```bash
   conda env create -f environment.yaml
   conda activate EGAPloc
   ```

3. **Download datasets**
   
   - **Vislocas Dataset**: [Zenodo](https://doi.org/10.5281/zenodo.10632698)
   - **HPA18 Dataset**: [GraphLoc](http://www.csbio.sjtu.edu.cn/bioinf/GraphLoc)

4. **Download pre-trained models** (optional)
   
   - **Vislocas Model**: [Download Link](https://jxstnueducn-my.sharepoint.com/:f:/g/personal/wanboyang_jxstnu_edu_cn/EpEDB3GcXMZFvRz9lQaBHswBYTEWUDF6ThPBHWqEPB-eUQ?e=jsSoY0)
   - **HPA18 Model**: Same repository as above

## 🏗️ Model Architecture

EGA-Ploc integrates three key components:

1. **Efficient Global-Local Attention**: Novel linear attention mechanism for computational efficiency
2. **Multi-scale Feature Fusion**: Hierarchical architecture capturing both local and global patterns
3. **Balanced Multi-label Learning**: Enhanced objective function addressing class imbalance

### Model Variants

- **ETP_cls_l1/l2/l3**: Base model variants with different backbone complexities
- **ETP_cls_cl0/cl1/cl2/cl3**: Cascaded backbone variants
- **Feature Fusion Models**: Multi-scale feature integration variants (featureAdd234, featureAdd324, etc.)

## 📊 Performance

EGA-Ploc achieves state-of-the-art performance on both Vislocas and HPA18 datasets, demonstrating superior accuracy in multi-label protein subcellular localization tasks.

## 📁 Project Structure

```
EGA-Ploc/
├── assets/                 # Sample images for demo testing
├── data/                   # Dataset annotation files (.csv)
├── datasets/               # Data loading modules
│   ├── ihc.py             # Vislocas dataset loader
│   ├── HPA18.py           # HPA18 dataset loader
│   ├── build.py           # Dataset builder
│   └── loader.py          # Data loader utilities
├── models/                 # Core model implementations
│   └── ETPLoc/            # EGA-Ploc model architecture
│       ├── backbone.py    # Backbone networks
│       ├── cls.py         # Classification heads
│       ├── nn/            # Neural network modules
│       └── utils/         # Model utilities
├── tools/                  # Training and testing scripts
│   ├── test.py            # Full dataset evaluation
│   ├── test_demo.py       # Single image testing
│   └── train.py           # Model training
├── utils/                  # Utility functions
│   ├── args.py            # Command-line arguments
│   ├── checkpoint.py      # Model checkpointing
│   ├── eval_metrics.py    # Performance evaluation
│   └── optimizer.py       # Optimization algorithms
└── results/               # Output directory for models and predictions
```

## 🛠️ Usage

### Testing on Full Datasets

```bash
# Test on Vislocas dataset
python tools/test.py --dataset IHC

# Test on HPA18 dataset  
python tools/test.py --dataset HPA18
```

### Single Image Testing

```bash
# Test a single image
python tools/test_demo.py --dataset IHC --single_image_path ./assets/Cytopl;Mito/55449_A_1_2.jpg
```

### Training

EGA-Ploc supports distributed training on multiple GPUs for efficient model training. The training process includes the following key components:

#### Training Setup

```bash
# Start distributed training with 8 GPUs
bash train.sh

# Or run directly with Python
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py
```

#### Training Configuration

The training process is configured through `utils/config_defaults.py` with the following key parameters:

- **Model Architecture**: Multiple EGA-Ploc variants including `AIP_discount_fa_4_cl1_3000_wd-005_mlce`
- **Training Epochs**: 120 epochs with warmup cosine scheduler
- **Learning Rate**: 5e-5 with AdamW optimizer
- **Batch Size**: 1 (due to high-resolution IHC images)
- **Loss Function**: Multi-label balanced cross-entropy to handle class imbalance
- **Regularization**: L1 and L2 regularization with configurable weights
- **Mixed Precision**: Enabled for memory efficiency

#### Training Process

1. **Data Loading**: 
   
   - Loads training and validation datasets from CSV annotation files
   - Supports both Vislocas and HPA18 datasets
   - Applies data augmentation for training set

2. **Model Initialization**:
   
   - Constructs EGA-Ploc classifier model
   - Supports distributed data parallel (DDP) training
   - Converts batch normalization to synchronized batch norm for multi-GPU training

3. **Optimization**:
   
   - **Optimizer**: AdamW with configurable weight decay (0.05, 0.01, 0.005, or 0)
   - **Scheduler**: Warmup cosine annealing scheduler
   - **Gradient Scaling**: Automatic mixed precision (AMP) for memory efficiency

4. **Loss Functions**:
   
   - **Multi-label Balanced Cross Entropy**: Handles class imbalance in protein localization
   - **BCE with Logits**: Standard binary cross-entropy option
   - **Multi-label Categorical Cross Entropy**: Alternative loss function

5. **Training Loop**:
   
   - Iterates through training data with periodic validation
   - Saves best model checkpoints based on validation loss
   - Supports checkpoint resumption for interrupted training
   - Logs training progress and metrics

6. **Evaluation**:
   
   - Periodic validation every 5 steps
   - Comprehensive evaluation metrics including precision, recall, F1-score
   - Multi-GPU synchronized evaluation

#### Supported Datasets

- **Vislocas**: 5 localization classes (cytoplasm, endoplasmic reticulum, mitochondria, nucleus, plasma membrane)
- **HPA18**: 7 localization classes (Cytoplasm, Golgi Apparatus, Mitochondria, Nucleus, Endoplasmic Reticulum, Plasma Membrane, Vesicles)

#### Model Checkpoints

Training automatically saves:

- **Latest Model**: For resuming interrupted training
- **Best Model**: Based on validation performance
- **Training Logs**: TensorBoard compatible logs for monitoring

To start training, ensure you have downloaded the required datasets and configured the appropriate data paths in the configuration files.

## 📋 Requirements

- **Platform**: Ubuntu 20.04+ (Windows/Linux supported)
- **GPU**: NVIDIA GPU with CUDA 11.3+
- **Python**: 3.8.15
- **PyTorch**: 1.11.0
- **Dependencies**: See `environment.yaml` for complete list

## 📄 License

### Academic Use

This project is released under the **Academic Free License v3.0** for non-commercial research and educational purposes. You are free to:

- ✅ Use, copy, and modify for academic research
- ✅ Share and distribute for educational purposes
- ✅ Build upon the work for non-commercial applications

### Commercial Use

For commercial licensing, please contact the authors to discuss terms and conditions. Commercial use requires explicit permission and may be subject to licensing fees.

## 🤝 Citation

If you use EGA-Ploc in your research, please cite our paper:

```bibtex
@article{wan2025ega,
  title={EGA-Ploc: An Efficient Global-Local Attention Model for Multi-label Protein Subcellular Localization Prediction on the Immunohistochemistry Images},
  author={Wan, Boyang and Huang, Xiaoyang and Qiao, Yang and Peng, Jiajie and Yang, Fan},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}
```

## 📞 Contact

For questions, issues, or collaboration opportunities:

- **Email**: `wanboyangjerry@163.com`
- **Issues**: [GitHub Issues](https://github.com/gd-hxy/EGA-Ploc/issues)

## 🙏 Acknowledgments

We thank the contributors and the research community for their valuable feedback and support in developing EGA-Ploc.

---

*EGA-Ploc: Advancing protein localization prediction through efficient deep learning architectures.*
