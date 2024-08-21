# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import inspect
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union

from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.layers import (
    ShapeSpec,
    cat,
    interpolate,
    nonzero_tuple,
    Conv2d,
    get_norm,
)
from detectron2.modeling import (
    ROI_HEADS_REGISTRY, 
    ROIHeads,
    build_box_head,
)
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.poolers_SAKD import ROIPooler_SAKD

from detectron2.modeling.roi_heads.roi_heads import (
    select_foreground_proposals, 
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

from detectron2.modeling.roi_heads.fast_rcnn_harmonize import FastRCNNOutputLayers_harmonize
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init

from lib.att_head import (
    build_att_head,
)

from lib.dis_head import (
    build_dis_head,
)

@ROI_HEADS_REGISTRY.register()
class ALLROIHeads_SAKD(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        dis_in_features: List[str],
        dis_pooler: ROIPooler_SAKD,
        # dis_head: nn.Module,

        train_on_pred_boxes: bool = False,

        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)

        self.dis_in_features = dis_in_features
        self.dis_pooler = dis_pooler

        self.train_on_pred_boxes = train_on_pred_boxes

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        if inspect.ismethod(cls._init_dis_head):
            ret.update(cls._init_dis_head(cfg, input_shape))

        return ret


    @classmethod
    def _init_dis_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        #pooler_resolution = [(19,19),(15,15),(11,11),(7,7)]
        pooler_resolution = [(19, 19), (19, 19), (19, 19), (19, 19)]
        #pooler_resolution = [(7, 7), (7, 7), (7, 7), (7, 7)]
        #pooler_resolution = [(15, 15), (15, 15), (15, 15), (15, 15)]
        #pooler_resolution = 15
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_DIS_HEAD.POOLER_SAMPLING_RATIO
        #pooler_type       = cfg.MODEL.ROI_DIS_HEAD.POOLER_TYPE
        pooler_type = "ROIAlign_SAKD"
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"dis_in_features": in_features}
        ret["dis_pooler"] = ROIPooler_SAKD(
            output_sizes=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # ret["dis_head"] = build_dis_head(
        #     cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        # )
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        gt_proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        alpha = 0,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        del targets

        if self.training:


            box_features, level_assignments = self._forward_dis(features, gt_proposals)
            teacher_box_features, _ = self._forward_dis(teacher_features, gt_proposals)

            del _

            return box_features, teacher_box_features, level_assignments

        else:
            pred_instances, _ = self._forward_box(features, gt_proposals, 0)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_att(features, instances)
        instances = self._forward_dis(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances], alpha
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features_haed = self.box_head(box_features)
        predictions = self.box_predictor(box_features_haed)
        del box_features_haed

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals, alpha)
            # # proposals is modified in-place below, so losses must be computed first.
            # if self.train_on_pred_boxes:
            #     with torch.no_grad():
            #         pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
            #             predictions, proposals
            #         )
            #         for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
            #             proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return predictions, box_features, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, _

    def _forward_att(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """

        features = [features[f] for f in self.att_in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            att_features = self.att_pooler(features, proposal_boxes)
            return self.att_head(att_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            att_features = self.att_pooler(features, pred_boxes)
            return self.att_head(att_features, instances)

    def _forward_dis(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """

        features = [features[f] for f in self.dis_in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            dis_features, level_assignments = self.dis_pooler(features, proposal_boxes)
            return dis_features, level_assignments
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            dis_features = self.dis_pooler(features, pred_boxes)
            return self.dis_head(dis_features, instances)