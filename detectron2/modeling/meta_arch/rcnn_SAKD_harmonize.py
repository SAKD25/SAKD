# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from typing import Dict, List, Optional, Tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

from .util_semckd import Embed, ConvReg, LinearEmbed, SelfA

from ..backbone import build_backbone, build_teacher_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads, build_roi_SAKD
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN_SAKD"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_SAKD(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.roi_SAKD = build_roi_SAKD(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.criterion = nn.MSELoss(reduction='none')
        self.criterion_box = nn.MSELoss()

    @property
    def device(self):
        return self.pixel_mean.device

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN_SAKD._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break

    def gt_logits(self, logits, inxs, gt_instances):

        gt_cls = [x.gt_classes for x in gt_instances]
        gt_cls = [x != -1 for x in gt_cls]
        gt_cls = [x[x] for x in gt_cls]
        gt_cls = [F.pad(input=x, pad=(0, inxs[i].size()[0] - x.size()[0]), mode='constant', value=0) for x,i in zip(gt_cls, range(len(gt_cls)))]
        inxs_mask = torch.cat(gt_cls)

        inxs = [x+4000*i for x,i in zip(inxs,range(len(inxs)))]
        inxs = torch.cat(inxs)

        inxs = inxs.reshape(-1, )
        inxs_mask = inxs_mask.reshape(-1, )

        inxs = inxs.argsort()

        inxs_mask = inxs_mask[inxs]

        logits = logits[0]
        logits = logits[inxs]

        return logits[inxs_mask]

    def gt_distance_logits(self, logits, inxs, gt_instances):

        gt_cls = [x.gt_classes for x in gt_instances]
        gt_cls = [x != -1 for x in gt_cls]
        gt_cls = [x[x] for x in gt_cls]
        gt_cls = [F.pad(input=x, pad=(0, inxs[i].size()[0] - x.size()[0]), mode='constant', value=0) for x, i in
                  zip(gt_cls, range(len(gt_cls)))]
        inxs_mask = torch.cat(gt_cls)


        inxs = [x + 4000 * i for x, i in zip(inxs, range(len(inxs)))]
        inxs = torch.cat(inxs)

        inxs = inxs.reshape(-1, )
        inxs_mask = inxs_mask.reshape(-1, )

        inxs = inxs.argsort()

        inxs_mask = inxs_mask[inxs]

        logits = logits['loss_dis']
        logits = logits[inxs]
        logits = logits.reshape(-1,)

        return logits[inxs_mask]

    def forward(self, batched_inputs, teacher_model:None, instance_mask:None, class_weight:None, self_attention:None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None


