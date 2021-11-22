import torch
import warnings

from mmaction.utils import import_module_error_func
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector3D

try:
    from mmdet.core import bbox2result
except (ImportError, ModuleNotFoundError):

    @import_module_error_func('mmdet')
    def bbox2result():
        pass

@DETECTORS.register_module()
class ViDETR(BaseDetector3D):
    """Base class for single-stage detectors.
    Single-stage detectors directly and sparsely predict bounding boxes on the
    output features (keyframe) of the backbone.
    """

    def __init__(self,
                 backbone,
                 obj_head=None,
                 act_head=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(ViDETR, self).__init__(init_cfg)
        
        self.backbone = build_backbone(backbone)
        self.obj_head = build_head(obj_head)
        self.act_head = build_head(act_head)

    def extract_feat(self, clip, keyframe):
        """Directly extract features from the backbone+neck."""
        obj_token, act_token = self.backbone(clip, keyframe)
        return obj_token, act_token

    def forward_dummy(self, clip, keyframe):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        obj_token, actor_token = self.extract_feat(clip, keyframe)
        obj_outs = self.obj_head(obj_token)
        act_outs = self.act_head(actor_token)
        return obj_outs, act_outs

    def forward_train(self,
                      clip,
                      keyframe,
                      clip_metas,
                      gt_obj_bboxes,
                      gt_obj_labels,
                      gt_act_bboxes,
                      gt_act_labels,
                      gt_obj_bboxes_ignore=None,
                      gt_act_bboxes_ignore=None):
        """
        Args:
            clip (Tensor): Input imgs of shape (N, C, T, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of img info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                keyframe in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(ViDETR, self).forward_train(clip=clip, keyframe=keyframe, clip_metas=clip_metas) # update clip_metas
        obj_token, act_token = self.extract_feat(clip, keyframe)

        losses_obj = self.obj_head.forward_train(obj_token, clip_metas,
                                              gt_obj_bboxes, gt_obj_labels,
                                              gt_obj_bboxes_ignore)
        
        (loss_obj_cls, loss_bbox, loss_iou) = losses_obj
        losses_act = self.act_head.forward_train(act_token, clip_metas,
                                              gt_act_bboxes, gt_act_labels,
                                              gt_act_bboxes_ignore)
        (loss_act_cls, loss_cls_metric) = losses_act
        # final_loss = loss_obj_cls + loss_bbox + loss_iou + loss_act_cls
        # return final_loss
        return dict({
            "loss_obj_cls": loss_obj_cls,
            "loss_bbox": loss_bbox,
            "loss_iou": loss_iou,
            "loss_act_cls": loss_act_cls
            })
    
    def forward_test(self, clips, keyframes, clip_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(clips, 'clips'), (keyframes, 'keyframes'), (clip_metas, 'clip_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(clips)
        if num_augs != len(keyframes):
            raise ValueError(f'num of augmentations ({len(clips)}) '
                             f'!= num of image meta ({len(clip_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for clip, clip_meta in zip(clips, clip_metas):
            batch_size = len(clip_meta)
            for clip_id in range(batch_size):
                clip_meta[clip_id]['batch_input_shape'] = tuple(clip.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(clips[0], keyframes[0], clip_metas[0], **kwargs)
        else:
            assert clips[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{clips[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(clips, keyframes, clip_metas, **kwargs)
        
    def simple_test(self, clip, keyframe, clip_metas, **kwargs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        obj_token, act_token = self.extract_feat(clip, keyframe)
        all_obj_cls_scores, all_bbox_preds = self.obj_head.forward(obj_token)
        all_act_cls_scores = self.act_head.forward(act_token)
        
        return all_act_cls_scores, all_bbox_preds
    
    def aug_test(self, clip, keyframe, clip_meta, **kwargs):
        """Test function with test time augmentation."""
        assert 'proposals' not in kwargs
        # TODO: support test augmentation for predefined proposals
        pass
    
    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of img metas ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError