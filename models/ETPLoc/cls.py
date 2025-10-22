"""
Classification module for EGA-Ploc protein subcellular localization / EGA-Ploc蛋白质亚细胞定位的分类模块
This module contains classification heads and model variants for multi-label protein localization.
此模块包含多标签蛋白质定位的分类头和模型变体。
"""

import math
import torch
import torch.nn as nn

from models.ETPLoc.backbone import ETPLargeBackbone, ETPCascadedBackbone
from models.ETPLoc.utils import build_kwargs_from_config
from typing import List, Dict
from .utils import list_sum
from .nn import (
    ConvLayer,
    LinearLayer,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
    UpSampleLayer,
)

__all__ = [
    "ClsNeck",
    "ETPCls",
    ######################
    "ETP_cls_l1",
    "ETP_cls_l2",
    "ETP_cls_l3",
    #####################
    "ETP_cls_cl0",
    "ETP_cls_cl1",
    "ETP_cls_cl2",
    "ETP_cls_cl3",
    #####################
    "ETP_cls_featureAdd234_cl1",
    "ETP_cls_featureAdd324_cl1",
    "ETP_cls_featureAdd432_cl1",
    "ETP_cls_featureAdd234_cl2",
    "ETP_cls_featureAdd324_cl2",
    "ETP_cls_featureAdd432_cl2",
    "ETP_cls_featureAdd324_cl3"
]


class ClsHead(OpSequential):
    def __init__(
        self,
        in_channels: int,
        width_list: List[int],
        n_classes=1000,
        dropout=0.0,
        norm="bn2d",
        act_func="hswish",
        fid="stage_final",
    ):
        ops = [
            ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
            nn.AdaptiveAvgPool2d(output_size=1),
            LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func),
            LinearLayer(width_list[1], n_classes, True, dropout, None, None),
        ]
        super().__init__(ops)

        self.fid = fid

    def forward(self, feed_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = feed_dict[self.fid]
        return OpSequential.forward(self, x)

class ClsNeck(nn.Module):
    def __init__(self,
                 fid_list: List[str],
                 in_channel_list: List[int],
                 head_width: int,
                 head_depth: int,
                 expand_ratio: float,
                 middle_op: str,
                 out_dim: List[int],
                 n_classes: int = 1000,
                 dropout=0.0,
                 first_stride=2,
                 upsample_ratio=1,
                 merge='add',
                 norm="bn2d",
                 act_func="gelu",
                 input_resolution=3000,
                 ):
        super().__init__()
        self.merge = merge
        self.head_depth = head_depth
        self.input_resolution = input_resolution
        self.first_stride = first_stride
        inputs = {}
        for fid, in_channel in zip(fid_list, in_channel_list):
            inputs[fid] = OpSequential(
                [
                    ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                    UpSampleLayer(size=math.ceil(input_resolution / (2 * self.first_stride * upsample_ratio))),
                ]
            )
        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        if self.merge == 'cat':
            self.cat_channel_ops = ConvLayer(head_width * 3, head_width, 1, norm=norm, act_func=None)
        if head_depth > 0:
            middle = []
            for _ in range(head_depth):
                if middle_op == "mb":
                    block = MBConv(
                        head_width,
                        head_width,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=(act_func, act_func, None),
                    )
                elif middle_op == "fmb":
                    block = FusedMBConv(
                        head_width,
                        head_width,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=(act_func, None),
                    )
                elif middle_op == "res":
                    block = ResBlock(
                        head_width,
                        head_width,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=(act_func, None),
                    )
                else:
                    raise NotImplementedError
                middle.append(ResidualBlock(block, IdentityLayer()))
            middle = OpSequential(middle)
            self.middle = middle

        outputs = [
            ConvLayer(head_width, out_dim[0], 1, norm=norm, act_func=act_func),
            nn.AdaptiveAvgPool2d(output_size=1),
            LinearLayer(out_dim[0], out_dim[1], False, norm="ln", act_func=act_func),
            LinearLayer(out_dim[1], n_classes, True, dropout, None, None),
        ]
        outputs = OpSequential(outputs)
        self.outputs = outputs

    def forward(self, feed_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        feat = [op(feed_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == 'cat':
            feat = torch.cat(feat, dim=1)
            feat = self.cat_channel_ops(feat)
        else:
            raise NotImplementedError
        if self.head_depth > 0:
            feat = self.middle(feat)
        feat = self.outputs(feat)
        return feat
    
    
class ETPCls(nn.Module):
    def __init__(self, backbone: ETPLargeBackbone or ETPCascadedBackbone, head: ClsHead or ClsNeck) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        output = self.head(feed_dict)
        return output

def ETP_cls_l1(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_l1

    backbone = ETP_backbone_l1(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = ETPCls(backbone, head)
    return model


def ETP_cls_l2(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_l2

    backbone = ETP_backbone_l2(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = ETPCls(backbone, head)
    return model


def ETP_cls_l3(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_l3

    backbone = ETP_backbone_l3(**kwargs)

    head = ClsHead(
        in_channels=1024,
        width_list=[6144, 6400],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = ETPCls(backbone, head)
    return model

def ETP_cls_cl0(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_cl0

    backbone = ETP_backbone_cl0(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[2304, 2560],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = ETPCls(backbone, head)
    return model

def ETP_cls_cl1(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_cl1

    backbone = ETP_backbone_cl1(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = ETPCls(backbone, head)
    return model

def ETP_cls_cl2(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_cl2

    backbone = ETP_backbone_cl2(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",

        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = ETPCls(backbone, head)
    return model

def ETP_cls_cl3(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_cl3

    backbone = ETP_backbone_cl3(**kwargs)

    head = ClsHead(
        in_channels=1024,
        width_list=[6144, 6400],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = ETPCls(backbone, head)
    return model

def ETP_cls_featureAdd234_cl1(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_cl1

    backbone = ETP_backbone_cl1(**kwargs)

    head = ClsNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmb",
        # first_stride=4,
        out_dim=[3072, 3200],
        **build_kwargs_from_config(kwargs, ClsNeck)
    )
    model = ETPCls(backbone, head)
    return model

def ETP_cls_featureAdd324_cl1(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_cl1

    backbone = ETP_backbone_cl1(**kwargs)

    head = ClsNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmb",
        # first_stride=4,
        upsample_ratio=2,
        out_dim=[3072, 3200],
        **build_kwargs_from_config(kwargs, ClsNeck)
    )
    model = ETPCls(backbone, head)
    return model

def ETP_cls_featureAdd432_cl1(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_cl1

    backbone = ETP_backbone_cl1(**kwargs)

    head = ClsNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmb",
        upsample_ratio=4,
        out_dim=[3072, 3200],
        **build_kwargs_from_config(kwargs, ClsNeck)
    )
    model = ETPCls(backbone, head)
    return model

def ETP_cls_featureAdd234_cl2(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_cl2

    backbone = ETP_backbone_cl2(**kwargs)

    head = ClsNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmb",
        # first_stride=4,
        out_dim=[3072, 3200],
        **build_kwargs_from_config(kwargs, ClsNeck)
    )
    model = ETPCls(backbone, head)
    return model

def ETP_cls_featureAdd324_cl2(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_cl2

    backbone = ETP_backbone_cl2(**kwargs)

    head = ClsNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmb",
        upsample_ratio=2,
        out_dim=[3072, 3200],
        **build_kwargs_from_config(kwargs, ClsNeck)
    )
    model = ETPCls(backbone, head)
    return model

def ETP_cls_featureAdd432_cl2(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_cl2

    backbone = ETP_backbone_cl2(**kwargs)

    head = ClsNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmb",
        upsample_ratio=4,
        out_dim=[3072, 3200],
        **build_kwargs_from_config(kwargs, ClsNeck)
    )
    model = ETPCls(backbone, head)
    return model

def ETP_cls_featureAdd324_cl3(**kwargs) -> ETPCls:
    from models.ETPLoc.backbone import ETP_backbone_cl3

    backbone = ETP_backbone_cl3(**kwargs)

    head = ClsNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=6,
        expand_ratio=4,
        middle_op="fmb",
        upsample_ratio=2,
        out_dim=[6144, 6400],
        **build_kwargs_from_config(kwargs, ClsNeck)
    )
    model = ETPCls(backbone, head)
    return model
