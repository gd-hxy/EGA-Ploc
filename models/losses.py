"""
Loss functions for EGA-Ploc protein subcellular localization / EGA-Ploc蛋白质亚细胞定位的损失函数
This module contains various loss functions for multi-label classification and contrastive learning.
此模块包含用于多标签分类和对比学习的各种损失函数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    Soft target cross entropy loss for multi-class classification with label smoothing.
    
    Formula:
    \[
    \mathcal{L} = -\sum_{i=1}^{C} y_i \log(\text{softmax}(x)_i)
    \]
    where \( C \) is the number of classes, \( y_i \) is the soft target probability.
    
    软目标交叉熵损失，用于带标签平滑的多分类任务。
    
    公式：
    \[
    \mathcal{L} = -\sum_{i=1}^{C} y_i \log(\text{softmax}(x)_i)
    \]
    其中 \( C \) 是类别数，\( y_i \) 是软目标概率。
    """

    def __init__(self, reduction="mean"):
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class NT_XentLoss(nn.Module):
    def __init__(self, norm=True, temperature=0.07, LARGE_NUM=1e10):
        super(NT_XentLoss, self).__init__()
        self.temperature = temperature
        self.LARGE_NUM = LARGE_NUM
        self.norm = norm

    def forward(self, out, dict_, idx, bag_id, condition):
        for k in range(len(bag_id)):
            new_bag_id = str(condition[k].item()) + "-" + str(bag_id[k].item())
            if new_bag_id in dict_:
                dict_[new_bag_id].update({idx[k].item(): out[k]})
            else:
                dict_.update({new_bag_id: {idx[k].item(): out[k]}})

        s_ = torch.tensor([]).to(device=out.device)
        labels_ = torch.tensor([]).to(device=out.device)
        l = []
        for i in range(len(bag_id)):
            new_bag_id_i = str(condition[i].item()) + "-" + str(bag_id[i].item())
            for j in range(i, len(bag_id)):
                new_bag_id_j = str(condition[j].item()) + "-" + str(bag_id[j].item())
                if new_bag_id_i == new_bag_id_j and i != j:
                    l.append(j)

        for i in range(len(bag_id)):
            if i in l:
                continue
            bag = torch.tensor([]).to(device=out.device)
            bag_label = torch.tensor([]).to(device=out.device)
            new_bag_id_i = str(condition[i].item()) + "-" + str(bag_id[i].item())
            for j in range(len(bag_id)):
                if j in l:
                    continue
                new_bag_id_j = str(condition[j].item()) + "-" + str(bag_id[j].item())
                a = torch.stack(list(dict_[new_bag_id_i].values()), axis=0)
                b = torch.stack(list(dict_[new_bag_id_j].values()), axis=0)
                logits_ = torch.matmul(a, b.transpose(0, 1)) / self.temperature

                if i == j:
                    masks = F.one_hot(torch.arange(0, len(dict_[new_bag_id_i])), num_classes=len(dict_[new_bag_id_i])).to(device=out.device)
                    logits_ = logits_ - masks * self.LARGE_NUM
                    ones = torch.ones_like(logits_).to(device=out.device)
                    diag = torch.diag_embed(torch.diag(ones))
                    bag_label = torch.cat((bag_label, ones - diag), dim=1)
                else:
                    bag_label = torch.cat((bag_label, torch.zeros_like(logits_).to(device=out.device)), dim=1)

                bag = torch.cat((bag, logits_), dim=1)
            s_ = torch.cat((s_, bag), dim=0)
            labels_ = torch.cat((labels_, bag_label), dim=0)

        loss = F.cross_entropy(s_, labels_)
        # loss = F.binary_cross_entropy_with_logits(s_, labels_)

        for k in range(len(bag_id)):
            new_bag_id = str(condition[k].item()) + "-" + str(bag_id[k].item())
            dict_.update({new_bag_id: {idx[k].item(): out[k].detach().data}})

        return loss


class Modified_NT_XentLoss(nn.Module):
    def __init__(self, temperature=1.0, LARGE_NUM=1e10, SMALL_NUM=1e-10):
        super(Modified_NT_XentLoss, self).__init__()
        self.temperature = temperature
        self.LARGE_NUM = LARGE_NUM
        self.SMALL_NUM = SMALL_NUM

    def forward(self, out, dict_, idx, bag_id, condition):
        for k in range(len(bag_id)):
            new_bag_id = str(condition[k].item()) + "-" + str(bag_id[k].item())
            if new_bag_id in dict_:
                dict_[new_bag_id].update({idx[k].item(): out[k]})
            else:
                dict_.update({new_bag_id: {idx[k].item(): out[k]}})

        s_ = torch.tensor([]).to(device=out.device)
        labels_ = torch.tensor([]).to(device=out.device)
        l = []
        for i in range(len(bag_id)):
            new_bag_id_i = str(condition[i].item()) + "-" + str(bag_id[i].item())
            for j in range(i, len(bag_id)):
                new_bag_id_j = str(condition[j].item()) + "-" + str(bag_id[j].item())
                if new_bag_id_i == new_bag_id_j and i != j:
                    l.append(j)

        for i in range(len(bag_id)):
            if i in l:
                continue
            bag = torch.tensor([]).to(device=out.device)
            bag_label = torch.tensor([]).to(device=out.device)
            new_bag_id_i = str(condition[i].item()) + "-" + str(bag_id[i].item())
            for j in range(len(bag_id)):
                if j in l:
                    continue
                new_bag_id_j = str(condition[j].item()) + "-" + str(bag_id[j].item())
                a = torch.stack(list(dict_[new_bag_id_i].values()), axis=0)
                b = torch.stack(list(dict_[new_bag_id_j].values()), axis=0)
                logits_ = torch.matmul(a, b.transpose(0, 1)) / self.temperature

                if i == j:
                    masks = F.one_hot(torch.arange(0, len(dict_[new_bag_id_i])), num_classes=len(dict_[new_bag_id_i])).to(device=out.device)
                    logits_ = logits_ - masks * self.LARGE_NUM
                    ones = torch.ones_like(logits_).to(device=out.device)
                    diag = torch.diag_embed(torch.diag(ones))
                    bag_label = torch.cat((bag_label, ones - diag), dim=1)
                else:
                    bag_label = torch.cat((bag_label, torch.zeros_like(logits_).to(device=out.device)), dim=1)

                exp_ = torch.exp(logits_)
                bag = torch.cat((bag, exp_), dim=1)
            s_ = torch.cat((s_, bag), dim=0)
            labels_ = torch.cat((labels_, bag_label), dim=0)

        exp_positive = s_ * labels_

        exp_div = torch.div(torch.sum(exp_positive, dim=1), torch.sum(s_, dim=1)) + self.SMALL_NUM
        loss = torch.mean(-torch.log(exp_div))

        for k in range(len(bag_id)):
            new_bag_id = str(condition[k].item()) + "-" + str(bag_id[k].item())
            dict_.update({new_bag_id: {idx[k].item(): out[k].detach().data}})

        return loss


class InfoNceLoss(nn.Module):
    def __init__(self, temperature=0.07, LARGE_NUM=1e12):
        super(InfoNceLoss, self).__init__()
        self.temperature = temperature
        self.LARGE_NUM = LARGE_NUM

    def forward(self, out, dict_, idx, bag_id, condition, batch_bag_id, batch_condition):
        s_ = torch.tensor([]).to(device=out.device)
        labels_ = torch.tensor([]).to(device=out.device)
        l = []
        for i in range(len(batch_bag_id)):
            new_bag_id_i = str(batch_condition[i].item()) + "-" + str(batch_bag_id[i].item())
            for j in range(i, len(batch_bag_id)):
                new_bag_id_j = str(batch_condition[j].item()) + "-" + str(batch_bag_id[j].item())
                if new_bag_id_i == new_bag_id_j and i != j:
                    l.append(j)

        for i in range(len(bag_id)):
            bag = torch.tensor([]).to(device=out.device)
            bag_label = torch.tensor([]).to(device=out.device)
            new_bag_id_i = str(condition[i].item()) + "-" + str(bag_id[i].item())
            for j in range(len(batch_bag_id)):
                if j in l:
                    continue
                new_bag_id_j = str(batch_condition[j].item()) + "-" + str(batch_bag_id[j].item())
                query = out[i].unsqueeze(0)
                key = torch.stack(list(dict_[new_bag_id_j].values()), axis=0)
                logits_ = torch.matmul(query, key.transpose(0, 1)) / self.temperature

                if new_bag_id_i == new_bag_id_j:
                    index = list(dict_[new_bag_id_j].keys()).index(idx[i].item())
                    ones = torch.ones_like(logits_).to(device=out.device)
                    logits_[0, index] -= self.LARGE_NUM
                    bag_label = torch.cat((bag_label, ones), dim=1)
                else:
                    bag_label = torch.cat((bag_label, torch.zeros_like(logits_).to(device=out.device)), dim=1)

                bag = torch.cat((bag, logits_), dim=1)
            s_ = torch.cat((s_, bag), dim=0)
            labels_ = torch.cat((labels_, bag_label), dim=0)

        exp_positive = s_ * labels_
        expsum_p = torch.logsumexp(exp_positive, dim=1)
        num_p = torch.sum(labels_, dim=1)
        num_p = torch.where(num_p > 1, num_p - 1, num_p)
        expsum_total = torch.logsumexp(s_, dim=1)

        loss = torch.mean(-expsum_p + torch.log(num_p) + expsum_total)

        return loss


class LogitsFocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    Addresses class imbalance by down-weighting easy examples.
    
    Formula:
    \[
    \text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
    \]
    where \( p_t = \exp(-\text{CE}(x, y)) \) is the predicted probability.
    
    多分类任务的Focal Loss，通过降低简单样本的权重来解决类别不平衡问题。
    
    公式：
    \[
    \text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
    \]
    其中 \( p_t = \exp(-\text{CE}(x, y)) \) 是预测概率。
    """
    def __init__(self, weight=None, reduction='mean', gamma=2, eps=1e-7):
        super(LogitsFocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Binary focal loss for multi-label classification with alpha balancing.
    Alpha-balanced focal loss for handling class imbalance.
    
    Formula:
    \[
    \text{FL} = -\alpha (1 - p_t)^\gamma y \log(p_t) - (1 - \alpha) p_t^\gamma (1 - y) \log(1 - p_t)
    \]
    where \( p_t = \sigma(x) \) is the sigmoid probability.
    
    带alpha平衡的多标签分类二元Focal Loss，用于处理类别不平衡。
    
    公式：
    \[
    \text{FL} = -\alpha (1 - p_t)^\gamma y \log(p_t) - (1 - \alpha) p_t^\gamma (1 - y) \log(1 - p_t)
    \]
    其中 \( p_t = \sigma(x) \) 是sigmoid概率。
    """
    def __init__(self, reduction='mean', alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, target):
        pt = torch.sigmoid(inputs)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class MultilabelCategoricalCrossEntropy(nn.Module):
    """
    Multilabel categorical cross entropy loss.
    Extension of cross entropy to multi-label classification with sigmoid activation.
    
    Formula:
    \[
    \mathcal{L} = \log\left(\sum_{i \in \text{neg}} \exp(x_i)\right) + \log\left(\sum_{i \in \text{pos}} \exp(-x_i)\right)
    \]
    where neg and pos are negative and positive classes respectively.
    
    多标签分类交叉熵损失，将交叉熵扩展到多标签分类任务。
    
    公式：
    \[
    \mathcal{L} = \log\left(\sum_{i \in \text{neg}} \exp(x_i)\right) + \log\left(\sum_{i \in \text{pos}} \exp(-x_i)\right)
    \]
    其中neg和pos分别表示负类和正类。
    """
    def __init__(self, reduction='mean', weight=None, pos_weight=None, LARGE_NUM=1e10):
        super(MultilabelCategoricalCrossEntropy, self).__init__()
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.LARGE_NUM = LARGE_NUM

    def forward(self, inputs, target):
        pt = torch.sigmoid(inputs)

        pred = (1 - 2 * target) * inputs
        pred_neg = pred - target * self.LARGE_NUM
        pred_pos = pred - (1 - target) * self.LARGE_NUM
        zeros = torch.zeros_like(pred[..., :1])

        pred_neg = torch.cat([pred_neg, zeros], axis=-1)
        pred_pos = torch.cat([pred_pos, zeros], axis=-1)

        neg_exp = torch.exp(pred_neg)
        pos_exp = torch.exp(pred_pos)

        if self.weight is not None:
            weight = torch.cat([self.weight, torch.ones_like(self.weight[:1])], axis=-1)
            neg_exp = neg_exp * weight
            pos_exp = pos_exp * weight
        if self.pos_weight is not None:
            neg_weight = torch.cat([torch.sqrt(1 / self.pos_weight), torch.ones_like(self.pos_weight[:1])], axis=-1)
            pos_weight = torch.cat([self.pos_weight, torch.ones_like(self.pos_weight[:1])], axis=-1)
            neg_exp = neg_exp * neg_weight
            pos_exp = pos_exp * pos_weight
        neg_sum = torch.sum(neg_exp, dim=1)
        pos_sum = torch.sum(pos_exp, dim=1)
        neg_loss = torch.log(neg_sum)
        pos_loss = torch.log(pos_sum)

        loss = neg_loss + pos_loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class MultilabelBalancedCrossEntropy(nn.Module):
    """
    Balanced multilabel cross entropy loss with class weighting.
    Handles class imbalance by weighting based on class frequencies.
    
    Formula:
    \[
    \mathcal{L} = \log\sum\exp(\text{pred\_neg}) + \log\sum\exp(\text{pred\_pos})
    \]
    with class balancing weight:
    \[
    w = \frac{\text{total\_nums}}{\sum(\text{target} \times \text{nums})}
    \]
    
    带类别加权的平衡多标签交叉熵损失，通过类别频率进行加权处理类别不平衡。
    
    公式：
    \[
    \mathcal{L} = \log\sum\exp(\text{pred\_neg}) + \log\sum\exp(\text{pred\_pos})
    \]
    带类别平衡权重：
    \[
    w = \frac{\text{total\_nums}}{\sum(\text{target} \times \text{nums})}
    \]
    """
    def __init__(self, reduction='mean', nums=None, total_nums=0, LARGE_NUM=1e12):
        super(MultilabelBalancedCrossEntropy, self).__init__()
        self.reduction = reduction
        self.total_nums = total_nums
        if total_nums == 0 and nums is not None:
            self.total_nums = torch.sum(nums)
        self.nums_sum = 0
        if nums is not None:
            self.nums_sum = torch.sum(nums)
        self.nums = nums
        self.LARGE_NUM = LARGE_NUM

    def forward(self, inputs, target):
        pred = (1 - 2 * target) * inputs
        pred_neg = pred - target * self.LARGE_NUM
        pred_pos = pred - (1 - target) * self.LARGE_NUM
        # print("pred: {}, pred_neg: {}, pred_pos: {}".format(pred, pred_neg, pred_pos))
        zeros = torch.zeros_like(pred[..., :1])

        pred_neg = torch.cat([pred_neg, zeros], axis=-1)
        pred_pos = torch.cat([pred_pos, zeros], axis=-1)
        # print("loss before logsumexp: neg: {}, pos: {}".format(pred_neg, pred_pos))
        # 参数截断，防止在logsumexp计算时产生一个很小的数值，导致出现数值下溢产生NaN
        # pred_neg = torch.clamp(pred_neg, min=1e-7, max=1 - 1e-7)
        # pred_pos = torch.clamp(pred_pos, min=1e-7, max=1 - 1e-7)

        neg_loss = torch.logsumexp(pred_neg, axis=-1)
        pos_loss = torch.logsumexp(pred_pos, axis=-1)
        # print("loss after exp: neg: {}, pos: {}".format(neg_loss, pos_loss))

        loss = neg_loss + pos_loss
        if self.nums is not None:
            nums = target * self.nums
            nums[nums == 0] = nums.max()
            loss = loss * (self.nums_sum / torch.sum(target * self.nums, dim=1))  # mean

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


_LOSSES = {
    "mae": nn.L1Loss,
    "mse": nn.MSELoss,
    "huber": nn.SmoothL1Loss,

    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "multi_label_soft_margin": nn.MultiLabelSoftMarginLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "focal": LogitsFocalLoss,
    "focal_loss": FocalLoss,
    "asl": AsymmetricLoss,
    "asl_optimized": AsymmetricLossOptimized,
    "multilabel_categorical_cross_entropy": MultilabelCategoricalCrossEntropy,
    "multilabel_balanced_cross_entropy": MultilabelBalancedCrossEntropy,

    "nt_xent": NT_XentLoss,
    "modified_nt_xent": Modified_NT_XentLoss,
    "info_nce": InfoNceLoss,
}


def get_loss_func(loss_name):
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


def l1_regularization(model, l1_alpha):
    if l1_alpha == 0:
        return 0
    l1_loss = 0
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss += torch.sum(abs(param))
    return l1_alpha * l1_loss

def l2_regularization(model, l2_alpha):
    if l2_alpha == 0:
        return 0
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)

def elasticnet_regularization(model, alpha, l1_ratio=0.5):
    return l1_regularization(model, alpha) * (1 - l1_ratio) + l2_regularization(model, alpha)
