"""
Classifier model factory for EGA-Ploc / EGA-Ploc的分类器模型工厂
This module provides a factory function to create different classifier models for protein subcellular localization.
此模块提供工厂函数来创建用于蛋白质亚细胞定位的不同分类器模型。
"""

import timm
from torchsummary import summary
from models.ETPLoc.cls_model_zoo import *


def getClassifier(cfg, model_name="ETP_fa_4_cl1_3000_wd-005_mlce"):
    if "ETP_fa_1" in model_name or "ETP_fa_2" in model_name or "ETP_fa_4" in model_name:
        m_name = model_name.split("_")[1] + '_' + model_name.split("_")[2] + '_' + model_name.split("_")[3]
        input_resolution = int(model_name.split("_")[4])
        model = create_cls_model(name=m_name, dropout=0.1, n_classes=cfg.CLASSIFIER.CLASSES_NUM,
                                 input_resolution=input_resolution)
    elif "ETP_discount_fa_1" in model_name or "ETP_discount_fa_2" in model_name or "ETP_discount_fa_4" in model_name:
        m_name = model_name.split("_")[2] + '_' + model_name.split("_")[3] + '_' + model_name.split("_")[4]
        input_resolution = int(model_name.split("_")[5])
        model = create_cls_model(name=m_name, dropout=0.1, param_discount=True, n_classes=cfg.CLASSIFIER.CLASSES_NUM,
                                 input_resolution=input_resolution)
    elif "ETP_" in model_name:
        m_name = model_name.split("_")[1]
        model = create_cls_model(name=m_name, dropout=0.1, n_classes=cfg.CLASSIFIER.CLASSES_NUM)
    else:
        model = create_cls_model(name="ETP_fa_4_cl1", dropout=0.1, n_classes=cfg.CLASSIFIER.CLASSES_NUM,
                                 input_resolution=3000)
    return model
