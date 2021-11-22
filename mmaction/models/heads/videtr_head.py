# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.losses import focal_loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from abc import ABCMeta, abstractmethod
from mmcls.models.utils import to_2tuple
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import (FFN, build_feedforward_network,
                                         build_positional_encoding)
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.core.bbox import bbox2roi
from mmdet.core.bbox.assigners import AssignResult, HungarianAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS as MMDET_BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs.builder import MATCH_COST as MMDET_MATCH_COST, build_match_cost
from mmdet.models import CrossEntropyLoss, binary_cross_entropy
from mmdet.models import build_loss as build_mmdet_loss
from mmdet.models import weight_reduce_loss
from mmdet.models.builder import LOSSES as MMDET_LOSSES
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.core.bbox.samplers import SamplingResult

from mmaction.core.bbox import bbox2result
from mmaction.utils import import_module_error_class
from ..builder import HEADS, MODELS

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

#TODO 预训练检测分支时，num_classes=训练集中标注类别，coco=80，ava=1；
#TODO 当然，最好用到ava数据集中的物体伪标签当作gt(alphaction的fasterrcnn跑的)，以提高person-obj行为类别的mAP
#TODO bg_class_weight为那些没有被分配到gt上的querys的损失权重，因此，检测分支总输出类别为 num_classes+1，最后一个类别指示 `not hook to gt`
#! https://github.com/open-mmlab/mmdetection/blob/6cf9aa1866b745fce8f1da6717fdb833d7c66fab/mmdet/models/dense_heads/detr_head.py

def recall_prec(pred_vec, target_vec):
    """
    Args:
        pred_vec (tensor[N x C]): each element is either 0 or 1
        target_vec (tensor[N x C]): each element is either 0 or 1

    """
    correct = pred_vec & target_vec
    # Seems torch 1.5 has no auto type conversion
    recall = correct.sum(1) / target_vec.sum(1).float()
    prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
    return recall.mean(), prec.mean()


def multi_label_accuracy(pred, target, thr=0.5, topk=(3, 5)):
    pred = pred.sigmoid()
    pred_vec = pred > thr
    # Target is 0 or 1, so using 0.5 as the borderline is OK
    target_vec = target > 0.5
    recall_thr, prec_thr = recall_prec(pred_vec, target_vec)

    recalls, precs = [], []
    for k in topk:
        _, pred_label = pred.topk(k, 1, True, True)
        pred_vec = pred.new_full(pred.size(), 0, dtype=torch.bool)

        num_sample = pred.shape[0]
        for i in range(num_sample):
            pred_vec[i, pred_label[i]] = 1
        recall_k, prec_k = recall_prec(pred_vec, target_vec)
        recalls.append(recall_k)
        precs.append(prec_k)
    return recall_thr, prec_thr, recalls, precs


def focal_loss_with_metrics(pred,
                            label,
                            focal_gamma=0.,
                            focal_alpha=1.,
                            weight=None,
                            reduction='mean',
                            avg_factor=None,
                            class_weight=None,
                            topk=(3, 5)):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, num_classes).
        label (torch.Tensor): The learning label of the prediction. (N, num_classes)
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss.
    """
    num_classes = label.size(1)
    if topk is None:
        topk = ()
    elif isinstance(topk, int):
        topk = (topk, )
    elif isinstance(topk, tuple):
        assert all([isinstance(k, int) for k in topk])
        topk = topk
    else:
        raise TypeError('topk should be int or tuple[int], '
                        f'but get {type(topk)}')
    #! Class 0 is ignored when calculaing multilabel accuracy,
    # so topk cannot be equal to num_classes
    assert all([k < num_classes for k in topk])

    class_weight = class_weight[1:]

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    label = label[:, 1:].float()
    pos_inds = torch.sum(label, dim=-1) > 0
    pred = pred[pos_inds, 1:]
    label = label[pos_inds]
    weight = weight[pos_inds]

    loss = F.binary_cross_entropy_with_logits(
        pred, label, pos_weight=class_weight, reduction='none')
    pt = torch.exp(-loss)
    F_loss = focal_alpha * (1 - pt)**focal_gamma * loss
    loss = torch.mean(F_loss, dim=1)

    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    metrics = dict()
    recall_thr, prec_thr, recall_k, prec_k = multi_label_accuracy(
        pred, label, thr=0.5, topk=topk)
    metrics['recall@thr=0.5'] = recall_thr
    metrics['prec@thr=0.5'] = prec_thr
    for i, k in enumerate(topk):
        metrics[f'recall@top{k}'] = recall_k[i]
        metrics[f'prec@top{k}'] = prec_k[i]

    return loss, metrics


def pseudo_sample_actlabel(assign_result, pseudo_bboxes=None, pseudo_gt_bboxes=None):
    """Directly returns the positive and negative indices  of samples.

    Args:
        assign_result (:obj:`AssignResult`): Assigned results
        bboxes (torch.Tensor): Bounding boxes
        gt_bboxes (torch.Tensor): Ground truth boxes

    Returns:
        :obj:`SamplingResult`: sampler results
    """
    num_query = assign_result.gt_inds.size(0)
        
    pos_inds = torch.nonzero(
        assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
    neg_inds = torch.nonzero(
        assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()

    num_gt = pos_inds.size(0)

    if pseudo_bboxes is None:
        pseudo_bboxes = assign_result.gt_inds.new_full((num_query, 4), -1, dtype=torch.float32)

    if pseudo_gt_bboxes is None:
        pseudo_gt_bboxes = assign_result.gt_inds.new_full((num_gt, 4), -1, dtype=torch.float32)

    gt_flags = pseudo_bboxes.new_zeros(pseudo_bboxes.shape[0], dtype=torch.uint8)
    
    sampling_result = SamplingResult(pos_inds, neg_inds, pseudo_bboxes, pseudo_gt_bboxes,
                                        assign_result, gt_flags)
    return sampling_result


@MMDET_LOSSES.register_module()
class MultiLabelFocalLoss(CrossEntropyLoss):

    def __init__(
        self,
        use_sigmoid=False,
        use_mask=False,
        reduction='mean',
        class_weight=None,
        ignore_index=None,
        loss_weight=1,
        focal_gamma=0.,
        focal_alpha=1.,
    ):
        super().__init__(
            use_sigmoid=use_sigmoid,
            use_mask=use_mask,
            reduction=reduction,
            class_weight=class_weight,
            ignore_index=ignore_index,
            loss_weight=loss_weight)

        self.cls_criterion = focal_loss_with_metrics
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        assert ignore_index is None, "`ignore_index` must be `None` currently"
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None

        loss_cls, metrics_cls = self.cls_criterion(
            cls_score,
            label,
            focal_gamma=self.focal_gamma,
            focal_alpha=self.focal_alpha,
            weight=weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        assert isinstance(metrics_cls, dict) 

        return loss_cls * self.loss_weight, metrics_cls


@MMDET_BBOX_ASSIGNERS.register_module()
class HungarianAssignerForObjectDetetion(HungarianAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0)):
        super().__init__(
            cls_cost=cls_cost, reg_cost=reg_cost, iou_cost=iou_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               ):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w = img_meta['keyframe_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels.to(torch.int64))
        # regression L1 cost
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)


@MMDET_MATCH_COST.register_module()
class MultiClassificationCost:
    """MultiClassificationCost.

     Args:
         weight (int | float, optional): loss_weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import \
         ... ClassificationCost
         >>> import torch
         >>> self = ClassificationCost()
         >>> cls_pred = torch.randn(4, 3) # [n_objs, n_classes]
         >>> gt_labels = (torch.randn(2, 3) > 0.8).type(torch.float32) # [n_objs, n_classes]
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.1683, -0.6500],
                [-0.4441, -0.3956],
                [-0.6208, -0.3514],
                [-0.6525, -0.2357]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, binary matrix, shape (num_gt, num_class).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # select and sum preds where gt=1,
        # suppouse that there are 2 gt for every query, therefore 2 multilabel cost for every query
        # [num_querys, num_classes] dot [num_classes, num_gts](binary matrix) --> [num_querys, num_gts]
        cls_score = cls_pred.softmax(-1)
        avg_factor = 1 / gt_labels.sum(dim=1) # [num_gts, 1]: num_gt_labels for each gt
        cls_cost = -torch.matmul(cls_score, gt_labels.T) * avg_factor
        return cls_cost * self.weight


@MMDET_BBOX_ASSIGNERS.register_module()
class HungarianAssignerForActorDetetion:
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_cost: (ConfigDict)
    """

    def __init__(self,
                 cls_cost=dict(type='MultiClassificationCost', weight=1.)):
        self.cls_cost = build_match_cost(cls_cost)

    def assign(
        self,
        cls_pred,
        gt_labels,
    ):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_classes].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt, num_classes).

        Returns:
            :obj:`AssignResult`: The assigned result.
                1.num_gts: (int)
                2.assigned_gt_inds: (tensor(torch.long)) [num_query,]
                3.max_overlaps: None
                4.labels: (tensor(torch.float32)) [num_query, num_classes] assigned multi-labels
        """
        num_gts, num_query, num_classes = gt_labels.size(0), cls_pred.size(
            0), gt_labels.size(1)
        # 1. assign -1 by default
        assigned_gt_inds = cls_pred.new_full((num_query, ),
                                             -1,
                                             dtype=torch.long)
        assigned_labels = cls_pred.new_full((num_query, num_classes),
                                            -1,
                                            dtype=torch.float32)
        if num_gts == 0 or num_query == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the cls costs
        # classification cost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        cost = cls_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            cls_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            cls_pred.device)

        # 4. assign backgrounds and foregrounds
        #! assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)


@HEADS.register_module()
class ViDETRObjHead(AnchorFreeHead):
    """#TODO Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_classes,
                 embed_dims=768,
                 num_obj_token=100,
                 sync_cls_avg_factor=False,
                 reg_ffn=dict(
                     type='FFN',
                     embed_dims=768,
                     feedforward_channels=768,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                     add_identity=False,
                 ),
                 act_cfg_after_reg_ffn=dict(type='ReLU', inplace=True),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssignerForObjectDetetion',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_obj_token = num_obj_token
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is ViDETRObjHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', )
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            #! set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.fp16_enabled = False

        self.loss_cls = build_mmdet_loss(loss_cls)
        self.loss_bbox = build_mmdet_loss(loss_bbox)
        self.loss_iou = build_mmdet_loss(loss_iou)

        self.activate = build_activation_layer(act_cfg_after_reg_ffn)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.reg_ffn = build_feedforward_network(reg_ffn)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.fc_reg = Linear(self.embed_dims, 4)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        pass

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None
                or version < 2) and self.__class__ is ViDETRObjHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, clip_metas, rescale=None):
        """Transform network output for a batch into bbox predictions.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        result_list = []
        for clip_id in range(len(clip_metas)):
            cls_score = cls_scores[clip_id]
            bbox_pred = bbox_preds[clip_id]
            img_shape = clip_metas[clip_id]['keyframe_shape']
            scale_factor = clip_metas[clip_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_obj_token)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      bbox_preds_list, gt_bboxes_list,
                                      gt_labels_list, img_metas,
                                      gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        #TODO 分配 num_query 个 bbox_pred 到 num_gt 个 gt_bboxes
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        #TODO 默认预测label的类别为 `背景类`, coco的背景类为: 类80,非背景类为0-79 ava的背景类为: 类1, 非背景类为0
        labels = gt_labels.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds] #! 直接将对应的gt_label给pred_label, 无比确定，连置信度都不用管的
        label_weights = gt_labels.new_ones(num_bboxes) #? 每一个label的权重都是1

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0 #! 仅正样本的bbox权重是1，其余为0 
        img_h, img_w = img_meta['keyframe_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
             clip_metas,
             gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            clip_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           clip_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)  # [num_querys, ]
        label_weights = torch.cat(label_weights_list, 0)  # [num_querys, ]
        bbox_targets = torch.cat(bbox_targets_list, 0) # [num_querys, 4]
        bbox_weights = torch.cat(bbox_weights_list, 0) # [num_querys, 4]

        # classification loss
        cls_scores = cls_scores.reshape(
            -1, self.cls_out_channels
        )  # [bs, num_query, cls_out_channels] --> [bs*num_query, cls_out_channels]
        #! construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for clip_meta, bbox_pred in zip(clip_metas, bbox_preds):
            img_h, img_w = clip_meta['keyframe_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def forward(self, obj_token):
        """"Forward function for a single feature level.
        Args:
            obj_token (Tensor): Input object tokens from backbone, shape
                [bs, n, c].
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        # [bs, num_query, embed_dim]

        all_cls_scores = self.fc_cls(obj_token)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(obj_token))).sigmoid()

        return all_cls_scores, all_bbox_preds

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      obj_token,
                      clip_metas,
                      gt_bboxes_list,
                      gt_labels_list=None,
                      gt_bboxes_ignore_list=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            obj_token (Tensor): Input obj_token of shape (bs, n, c).
            clip_metas (list[dict]): Meta information of each clip, e.g.,
                clip space shape, scaling factor, etc.
            gt_bboxes_list (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore_list (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'

        outs = self(obj_token)
        if gt_bboxes_list is None:
            loss_inputs = outs + (gt_bboxes_list, clip_metas)
        else:
            loss_inputs = outs + (gt_bboxes_list, gt_labels_list, clip_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore_list=gt_bboxes_ignore_list)
        return losses


@HEADS.register_module()
class ViDETRActHead(AnchorFreeHead):
    """#TODO Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(
            self,
            num_classes,
            embed_dims=768,
            num_actor_token=100,
            sync_cls_avg_factor=False,
            reg_ffn=dict(
                type='FFN',
                embed_dims=768,
                feedforward_channels=768,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='ReLU', inplace=True),
                add_identity=False,
            ),
            act_cfg_after_reg_ffn=dict(type='ReLU', inplace=True),
            loss_cls=dict(
                type='MultiLabelFocalLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0,
                focal_gamma=0.,
                focal_alpha=1.,
            ),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssignerForActorDetetion',
                    cls_cost=dict(type='MultiClassificationCost', weight=1.),
                )),
            test_cfg=dict(max_per_img=100),
            init_cfg=None,
            **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_actor_token = num_actor_token
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is ViDETRActHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes) * class_weight
            #! set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            self.assigner = build_assigner(assigner)
            #! ViDETR sampling=False, so use PseudoSampler
            self.sampler = pseudo_sample_actlabel
            

        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.fp16_enabled = False

        self.loss_cls = build_mmdet_loss(loss_cls)

        self.activate = build_activation_layer(act_cfg_after_reg_ffn)

        self.cls_out_channels = num_classes

        self.reg_ffn = build_feedforward_network(reg_ffn)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.fc_reg = Linear(self.embed_dims, 4)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        pass

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None
                or version < 2) and self.__class__ is ViDETRActHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, clip_metas, rescale=None):
        """Transform network output for a batch into bbox predictions.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        result_list = []
        for clip_id in range(len(clip_metas)):
            cls_score = cls_scores[clip_id]
            bbox_pred = bbox_preds[clip_id]
            img_shape = clip_metas[clip_id]['keyframe_shape']
            scale_factor = clip_metas[clip_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_actor_token)
        #! exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels

    def get_targets(
        self,
        cls_scores_list,
        gt_labels_list,
        img_metas,
    ):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        (labels_list, label_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list, gt_labels_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(
        self,
        cls_score,
        gt_labels,
    ):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, num_clasees). 

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_query, num_classes = cls_score.size(0), cls_score.size(1)
        # assigner and sampler
        assign_result = self.assigner.assign(cls_score, gt_labels)
        sampling_result = self.sampler(assign_result)  
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = cls_score.new_full(
            (num_query, num_classes),
            num_classes,
            dtype=torch.float32)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(num_query)

        return (labels, label_weights, pos_inds, neg_inds)

    def loss(
        self,
        cls_scores,
        gt_labels_list,
        clip_metas,
    ):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            clip_metas (list[dict]): List of image meta information.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, gt_labels_list,
                                           clip_metas)
        (labels_list, label_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls, loss_cls_metric = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        return loss_cls, loss_cls_metric

    def forward(self, act_token):
        """"Forward function for a single feature level.
        Args:
            act_token (Tensor): Input object tokens from backbone, shape
                [bs, n, c].
            clip_metas (list[dict]): List of clip information.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        # [bs, num_query, embed_dim]

        all_cls_scores = self.fc_cls(act_token)
        # all_bbox_preds = self.fc_reg(self.activate(
        #     self.reg_ffn(act_token))).sigmoid()

        return all_cls_scores

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      act_token,
                      clip_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore_list=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            keyframe (Tensor): Input keyframe of shape (N, C, H, W).
            clip_metas (list[dict]): Meta information of each clip, e.g.,
                clip space shape, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            #! baseline 不需要在此head监督actor坐标, 已经在obj_head完成纯keyframe上的监督
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore_list (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'

        outs = self(act_token) # all_cls_scores [num_query,]
        #? 0. actorbbox用哪些输入可以得到, actor_bbox损失和obj_bbox损失如何权衡? 
        #? 1. baseline: 不加入obj位置的作为先验, 仅使用人做监督, objhead输出的 obj_4pos==actor_4pos
        #? 2. 如果监督obj信息? ==> 此模型可以用作COCO目标检测; 
        #?    但做动作检测时, 仅需要将obj_tokens中认为是person的那些纳入计算图
        assert isinstance(outs, torch.Tensor)
        if gt_labels is None:
            loss_inputs = (outs,) + (clip_metas)
        else:
            loss_inputs = (outs,) + (gt_labels, clip_metas)
        loss_cls, loss_cls_metric = self.loss(*loss_inputs) # loss_cls, loss_cls_metric
        return loss_cls, loss_cls_metric
