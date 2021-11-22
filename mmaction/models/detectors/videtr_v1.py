import torch
import warnings

from ..builder import DETECTORS
from .single_stage import SingleStageDetector3D


@DETECTORS.register_module()
class ViDETR(SingleStageDetector3D):
    r"""#TODO Implementation of `ViDETR: End-to-End Video Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""

    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ViDETR, self).__init__(backbone, None, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    # over-write extract_feat beacause 
    def extract_feat(self, clip):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(clip)
        if self.with_neck:
            x = self.neck(x)
        return x

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    #TODO clip 需要特别指出 keyframe
    def forward_dummy(self, clip):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in ViDETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, T, height, width = clip.shape
        dummy_clip_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(clip)
        outs = self.bbox_head(x, dummy_clip_metas)
        return outs

    # over-write `onnx_export` because:
    # (1) the forward of bbox_head requires clip_metas
    # (2) the different behavior (e.g. construction of `masks`) between
    # torch and ONNX model, during the forward of bbox_head
    def onnx_export(self, clip, clip_metas):
        """Test function for exporting to ONNX, without test time augmentation.
        Args:
            clip (torch.Tensor): input images.
            clip_metas (list[dict]): List of image information.
        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(clip)
        # forward of this head requires clip_metas
        outs = self.bbox_head.forward_onnx(x, clip_metas)
        # get shape as tensor
        clip_shape = torch._shape_as_tensor(clip)[2:]
        clip_metas[0]['clip_shape_for_onnx'] = clip_shape

        det_bboxes, det_labels = self.bbox_head.onnx_export(*outs, clip_metas)

        return det_bboxes, det_labels
