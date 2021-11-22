# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmaction.models.detectors import SingleStageDetector3D, BaseDetector3D, ViDETR
from ..base import generate_videtr_demo_inputs, get_detector_cfg

try:
    from mmaction.models import build_detector
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


@pytest.mark.skipif(not mmdet_imported, reason='requires mmdet')
def test_ava_detector():
    config = get_detector_cfg('ava_exp/ava22_baseline.py')
    detector = build_detector(config.model)

    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            #! 嵌入 FasterRCNN时，设置single_stage=False
            train_demo_inputs = generate_videtr_demo_inputs(
                input_shape=(1, 3, 8, 224, 224),
                keyframe_shape=(1, 3, 800, 1344),
                num_act_classes=81,
                num_obj_classes=4,
                train=True,
                device='cuda',
                single_stage=True)
            test_demo_inputs = generate_videtr_demo_inputs(
                input_shape=(1, 3, 8, 224, 224),
                keyframe_shape=(1, 3, 800, 1344),
                num_act_classes=81,
                num_obj_classes=4,
                train=False,
                device='cuda',
                single_stage=True)
            detector = detector.cuda()

            losses = detector(**train_demo_inputs)
            assert isinstance(losses, dict)

            # Test forward test
            with torch.no_grad():
                _ = detector(**test_demo_inputs, return_loss=False)
    else:
        train_demo_inputs = generate_videtr_demo_inputs(
            input_shape=(1, 3, 8, 224, 224),
            keyframe_shape=(1, 3, 800, 1344),
            num_act_classes=81,
            num_obj_classes=4,
            train=True,
            single_stage=True)
        test_demo_inputs = generate_videtr_demo_inputs(
            input_shape=(1, 3, 8, 224, 224),
            keyframe_shape=(1, 3, 800, 1344),
            num_act_classes=81,
            num_obj_classes=4,
            train=False,
            single_stage=True)
        # losses = detector(**train_demo_inputs)
        # assert isinstance(losses, dict)

        # Test forward test
        with torch.no_grad():
            _ = detector(**test_demo_inputs, return_loss=False)
