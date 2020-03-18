# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 1:32:21
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:33:21
# from:
# https://github.com/naivelamb/kaggle-cloud-organization/blob/master/losses.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(-1, 1)
        target = target.view(-1, 1)

        pt = torch.sigmoid(input)
        pt = 1 - (pt - target.float()).abs()
        logpt = pt.log()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.long().data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()  # -*- coding: utf-8 -*-


class SoftDiceLoss(nn.Module):
    """Differentiable soft dice loss.

    Note: Sigmoid is automatically applied here!
    """
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        eps = 1e-9
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1).float()
        intersection = torch.sum(m1 * m2, 1)
        union = torch.sum(m1, dim=1) + torch.sum(m2, dim=1)
        score = (2*intersection + eps)/(union + eps)
        score = (1 - score).mean()
        return score


class WeightedBCE(nn.Module):
    def __init__(self, weights=None):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit, truth):
        batch_size, num_class = truth.shape
        logit = logit.view(batch_size, num_class)
        truth = truth.view(batch_size, num_class)
        assert(logit.shape == truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth,
                                                  reduction='none')

        if self.weights is None:
            loss = loss.mean()
        else:
            pos = (truth > 0.5).float()
            neg = (truth < 0.5).float()
            pos_sum = pos.sum().item() + 1e-12
            neg_sum = neg.sum().item() + 1e-12
            loss = (self.weights[1]*pos*loss/pos_sum +
                    self.weights[0]*neg*loss/neg_sum).sum()
        return loss


class MultiLabelDiceLoss(nn.Module):
    """The average dice across multiple classes.

    Note: Sigmoid is automatically applied here!
    """
    def __init__(self):
        super(MultiLabelDiceLoss, self).__init__()
        self.dice_loss = SoftDiceLoss()

    def forward(self, logits, targets):
        loss = 0
        num_classes = targets.size(1)
        for class_nr in range(num_classes):
            loss += self.dice_loss(logits[:, class_nr, :, :],
                                   targets[:, class_nr, :, :])
        return loss/num_classes


class ComboLoss(nn.Module):
    """Weighted classification and segmentation loss.

    Attributes:
        weights (list):
        activation:
        bce: with logits loss
        dice_loss: soft dice loss (all classes)

    """
    def __init__(self, weights=[0.1, 0, 1], activation=None):
        """
        Args:
            weights (list): [image_cls, pixel_seg, pixel_cls]
            activation: One of ['sigmoid', None]
        """
        super(ComboLoss, self).__init__()
        self.weights = weights
        self.activation = activation
        assert self.activation in ["sigmoid", None], \
            "`activation` must be one of ['sigmoid', None]."
        self.bce = nn.BCEWithLogitsLoss(reduce=True)
        self.dice_loss = MultiLabelDiceLoss()

    def create_fc_tensors(self, logits, targets):
        """Creates the classification tensors from the segmentation ones.
        """
        batch_size, num_classes, _, _ = targets.shape
        # gathers all of the slices that actually have an ROI
        summed = targets.view(batch_size, num_classes, -1).sum(-1)
        targets_fc = (summed > 0).float()
        # gathers all of the slices that have been predicted to have an ROI
        logits_fc = logits.view(batch_size, num_classes, -1)
        logits_fc = torch.max(logits_fc, -1)[0]
        return logits_fc, targets_fc

    def forward(self, logits, targets):
        # Classification tensors derived from the segmentation ones
        logits_fc, targets_fc = self.create_fc_tensors(logits, targets)
        # Activation output
        p = F.sigmoid(logits) if self.activation == "sigmoid" else logits

        # Actual computing the losses
        if self.weights[0]:
            loss_fc = self.weights[0] * self.bce(logits_fc, targets_fc)
        else:
            loss_fc = torch.tensor(0).cuda()

        if self.weights[1] or self.weights[2]:
            loss_seg_dice = self.weights[1] * self.dice_loss(p, targets)
            # pixel cls
            loss_seg_bce = self.weights[2] * self.bce(logits, targets)
        else:
            loss_seg_dice = torch.tensor(0).cuda()
            loss_seg_bce = torch.tensor(0).cuda()

        loss = loss_fc + loss_seg_bce + loss_seg_dice
        return loss


class ComboLossOnlyPos(ComboLoss):
    """Weighted classification and segmentation loss. (Considers only positive)

    This loss is only calculated on labels with ROIs in them to maximize
    the foreground class prediction capability.

    Attributes:
        weights (list):
        activation:
        bce: with logits loss
        dice_loss: soft dice loss (per class)

    """
    def __init__(self, weights=[0.1, 0, 1], activation=None):
        """
        Args:
            weights (list): [image_cls, pixel_seg, pixel_cls]
            activation: One of ['sigmoid', None]
        """
        super().__init__(weights=weights, activation=activation)
        self.dice_loss = SoftDiceLoss()

    def forward(self, logits, targets):
        logits_fc, targets_fc = self.create_fc_tensors(logits, targets)
        # extracting the positive and negative class indices
        n_pos = targets_fc.sum()
        pos_idx = (targets_fc > 0.5)
        neg_idx = (targets_fc < 0.5)

        # Actual computing the losses
        if self.weights[0]:
            loss_fc = self.weights[0] * self.bce(logits_fc[neg_idx],
                                                 targets_fc[neg_idx])
        else:
            loss_fc = torch.tensor(0).cuda()

        if self.weights[1] or self.weights[2]:
            logits_pos, targets_pos = logits[pos_idx], targets[pos_idx]
            # pixel seg
            if n_pos == 0:
                loss_seg_dice = torch.tensor(0).cuda()
            else:
                loss_seg_dice = self.weights[1] * self.dice_loss(logits_pos,
                                                                 targets_pos)
            # pixel cls
            loss_seg_bce = self.weights[2] * self.bce(logits_pos, targets_pos)
        else:
            loss_seg_dice = torch.tensor(0).cuda()
            loss_seg_bce = torch.tensor(0).cuda()

        loss = loss_fc + loss_seg_bce + loss_seg_dice
        return loss
