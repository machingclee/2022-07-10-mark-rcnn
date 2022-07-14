from random import random
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from torchvision.transforms.functional import resize
from .detector import MLPDetector, MaskDetector
from .context_block import context_block2d
from .feature_extractor import FeatureExtractor
from .box_utils import (
    assign_gts_to_anchors_or_proposals, clip_boxes_to_image,
    decode_deltas_to_boxes, encode_boxes_to_deltas, remove_small_boxes,
    decode_single
)
from .rpn import RPN
from . import config
from typing import cast
from torch import Tensor
from torchvision.ops import nms
from .utils import random_choice, smooth_l1_loss
from .device import device
from torchvision.ops import roi_align
from PIL import Image, ImageDraw


cce_loss = nn.CrossEntropyLoss()


class MaskRCNN(nn.Module):
    def __init__(self, pred_score_thresh=config.pred_score_thresh):
        super(MaskRCNN, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.rpn = RPN().to(device)
        # self.ctx_blk = context_block2d(512).to(device)
        self.ctx_blk = self.identity
        self.mlp_detector = MLPDetector().to(device)
        self.mask_detector = MaskDetector().to(device)
        self.pred_score_thresh = pred_score_thresh
        
    def identity(self, x):
        return x

    def filter_small_rois(self, logits, rois):
        rois = rois.squeeze(1).unsqueeze(0)
        hs = rois[..., 2] - rois[..., 0]
        ws = rois[..., 3] - rois[..., 1]
        keep_mask = (hs >= config.min_size) * (ws >= config.min_size)
        logits = logits[keep_mask]
        rois = rois[keep_mask]
        return logits, rois

    def filter_by_nms(self, logits, rois, n_pre_nms, n_post_nms, thresh):
        scores = logits.softmax(dim=1)[:, 1]
        order = scores.ravel().argsort(descending=True)
        order = order[:n_pre_nms]
        scores = scores[order]
        rois = rois[order, :]
        keep = nms(rois, scores, thresh)
        keep = keep[:n_post_nms]
        logits = logits[keep]
        rois = rois[keep]
        return logits, rois

    def get_rpn_loss(self, labels, distributed_gt_boxes, pred_deltas, pred_fg_bg_logit):
        labels = labels.to(device)
        pos_mask = labels == 1
        keep_mask = torch.abs(labels) == 1

        target_deltas = encode_boxes_to_deltas(distributed_gt_boxes, self.rpn.anchors)
        objectness_label = torch.zeros_like(labels, device=device, dtype=torch.long)
        objectness_label[labels == 1] = 1.0
        if torch.sum(pos_mask) > 0:
            rpn_reg_loss = smooth_l1_loss(pred_deltas[pos_mask], target_deltas[pos_mask])
        else:
            rpn_reg_loss = None
        rpn_cls_loss = cce_loss(pred_fg_bg_logit.squeeze(0)[keep_mask], objectness_label[keep_mask])

        return rpn_cls_loss, rpn_reg_loss

    def get_roi_loss(self, labels, distributed_gt_boxes_to_roi, rois, pred_deltas, cls_logits, distributed_cls_idxes_to_roi):
        target_deltas = encode_boxes_to_deltas(
            distributed_gt_boxes_to_roi, rois, weights=config.roi_head_encode_weights
        )
        N = cls_logits.shape[0]
        pred_deltas = pred_deltas.reshape(N, -1, 4)

        target_deltas = target_deltas[labels == 1]
        keep_mask = torch.abs(labels) == 1
        sub_labels = labels[keep_mask]

        distributed_cls_idxes_to_roi = distributed_cls_idxes_to_roi[keep_mask]
        pos_idx = torch.where(sub_labels == 1)[0]
        neg_idx = torch.where(sub_labels == -1)[0]
        classes = distributed_cls_idxes_to_roi[pos_idx]
        
        if len(pos_idx) == 0:
            roi_reg_loss = pred_deltas.sum() * 0
        else:
            roi_reg_loss = smooth_l1_loss(
                target_deltas,
                pred_deltas[pos_idx, classes.long()]
            )
            
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        if n_pos > 0:
            roi_cls_loss = n_pos * cce_loss(cls_logits[pos_idx], distributed_cls_idxes_to_roi[pos_idx].long())
            roi_cls_loss += n_neg * cce_loss(cls_logits[neg_idx], distributed_cls_idxes_to_roi[neg_idx].long())
        else:
            roi_cls_loss = n_neg * cce_loss(cls_logits[neg_idx], distributed_cls_idxes_to_roi[neg_idx].long())
        roi_cls_loss = roi_cls_loss / (n_pos + n_neg)
        return roi_cls_loss, roi_reg_loss

    def filter_by_scores_and_size(self, cls_logits, pred_boxes, mask_logits):
        cls_idxes = torch.arange(config.n_classes, device=device)
        cls_idxes = cls_idxes[None, ...].expand_as(cls_logits)

        scores = cls_logits.softmax(dim=1)[:, 1:]
        boxes = pred_boxes[:, 1:]
        cls_idxes = cls_idxes[:, 1:]
        mask_logits = mask_logits[:, 1:]

        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        cls_idxes = cls_idxes.reshape(-1)
        mask_logits = mask_logits.reshape(-1, 28, 28)

        idxes = torch.where(torch.gt(scores, self.pred_score_thresh))[0]
        boxes = boxes[idxes]
        scores = scores[idxes]
        cls_idxes = cls_idxes[idxes]
        mask_logits = mask_logits[idxes]

        keep = remove_small_boxes(boxes, min_size=1)
        boxes = boxes[keep]
        scores = scores[keep]
        cls_idxes = cls_idxes[keep]
        mask_logits = mask_logits[keep]

        return scores, boxes, cls_idxes, mask_logits

    def get_mask_loss(
        self, 
        labels, 
        distributed_cls_indexes, 
        distributed_gt_mask_pooling_pred_targets_to_roi,
        mask_logits
    ):
        sub_labels = labels[torch.abs(labels) == 1]
        pos_idx = torch.where(sub_labels==1)[0]
        pos_cls_idxes = distributed_cls_indexes[labels==1]  
        
        gt_mask = distributed_gt_mask_pooling_pred_targets_to_roi     # reduce variable length
        
        if gt_mask.numel() == 0:
            return mask_logits.sum() * 0
        
        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[
                pos_idx, 
                pos_cls_idxes.long()
            ],
            gt_mask[labels==1]
        )
        
        return mask_loss
        
    def forward(
        self,
        x,
        gt_boxes=None,
        gt_cls_indexes=None,
        gt_mask_pooling_pred_targets=None
    ):
        x = x.to(device)

        if gt_boxes is not None:
            gt_boxes = gt_boxes.to(device)

        if gt_cls_indexes is not None:
            gt_cls_indexes = gt_cls_indexes.to(device)

        if self.training:
            assert gt_boxes is not None
            assert gt_cls_indexes is not None

        out_feat = self.feature_extractor(x)

        features = out_feat
        pred_fg_bg_logits, pred_deltas = self.rpn(features)
        pred_fg_bg_logits = pred_fg_bg_logits.squeeze(0)
        pred_deltas = pred_deltas.squeeze(0)

        if self.training:
            labels, distributed_gt_boxes, _, _ = \
            assign_gts_to_anchors_or_proposals(
                gt_boxes,
                gt_mask_pooling_pred_targets,
                self.rpn.anchors,
                n_sample=config.rpn_n_sample,
                pos_sample_ratio=config.rpn_pos_ratio,
                pos_iou_thresh=config.target_pos_iou_thres,
                neg_iou_thresh=config.target_neg_iou_thres,
                target_cls_indexes=None
            )
            
            rpn_cls_loss, rpn_reg_loss = self.get_rpn_loss(
                labels, distributed_gt_boxes, pred_deltas, pred_fg_bg_logits
            )

        rois = decode_deltas_to_boxes(pred_deltas.detach().clone(), self.rpn.anchors)
        rois = clip_boxes_to_image(rois)
        rois = rois.squeeze(0)

        if self.training:
            pred_fg_bg_logits, rois = self.filter_by_nms(
                pred_fg_bg_logits.detach().clone(),
                rois,
                config.n_train_pre_nms,
                config.n_train_post_nms,
                config.nms_train_iou_thresh
            )
            # distribute to anchors
            labels, distributed_gt_boxes_to_roi, distributed_gt_mask_pooling_pred_targets_to_roi, distributed_cls_idxes_to_roi = \
                assign_gts_to_anchors_or_proposals(
                    gt_boxes,
                    gt_mask_pooling_pred_targets,
                    rois,
                    n_sample=config.roi_n_sample,
                    pos_sample_ratio=config.roi_pos_ratio,
                    pos_iou_thresh=config.roi_pos_iou_thresh,
                    neg_iou_thresh=config.roi_neg_iou_thresh,
                    target_cls_indexes=gt_cls_indexes
                )

            roi_pooling = roi_align(
                out_feat,
                [rois[torch.abs(labels) == 1].mul_(1 / 16.0)],
                (7, 7),
            )
            mask_pooling = roi_align(
                out_feat,
                [rois[torch.abs(labels) == 1].mul_(1 / 16.0)],
                (14, 14),
            )
        else:
            pred_fg_bg_logits, rois = self.filter_by_nms(
                pred_fg_bg_logits.clone(),
                rois.clone(),
                config.n_eval_pre_nms,
                config.n_eval_post_nms,
                config.nms_eval_iou_thresh
            )

            pred_fg_bg_logits = pred_fg_bg_logits[:config.roi_n_sample]
            rois = rois[:config.roi_n_sample]

            roi_pooling = roi_align(
                out_feat,
                [rois.clone().mul_(1 / 16.0)],
                (7, 7)
            )
            mask_pooling = roi_align(
                out_feat,
                [rois.clone().mul_(1 / 16.0)],
                (14, 14),
            )
            
        roi_pooling = self.ctx_blk(roi_pooling)
        mask_pooling = self.ctx_blk(mask_pooling)
        
        cls_logits, roi_pred_deltas = self.mlp_detector(roi_pooling)
        mask_logits = self.mask_detector(mask_pooling)

        if self.training:
            roi_cls_loss, roi_reg_loss = self.get_roi_loss(
                labels,
                distributed_gt_boxes_to_roi,
                rois,
                roi_pred_deltas,
                cls_logits,
                distributed_cls_idxes_to_roi
            )
            
            mask_loss = self.get_mask_loss(
                labels,
                distributed_cls_idxes_to_roi, 
                distributed_gt_mask_pooling_pred_targets_to_roi, 
                mask_logits
            )
            
            return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss, mask_loss
        else:
            N = rois.shape[0]
            roi_pred_deltas = roi_pred_deltas.reshape(N, -1, 4)
            pred_boxes = decode_deltas_to_boxes(
                roi_pred_deltas, rois, weights=config.roi_head_encode_weights
            ).squeeze(0)

            scores, boxes, cls_idxes, mask_logits = self.filter_by_scores_and_size(cls_logits, pred_boxes, mask_logits)
            cls_idxes = cls_idxes - 1   # the output will ignore background class

            if boxes.numel() == 0:
                scores = []
                boxes = []
                cls_idxes = []
                mask = []
                return scores, boxes, cls_idxes, rois, mask

            keep = nms(boxes, scores, config.output_nms_within_class)

            scores = scores[keep]
            boxes = boxes[keep]
            cls_idxes = cls_idxes[keep]
            mask_logits = mask_logits[keep]
            masks = torch.where(torch.sigmoid(mask_logits) > 0.5, 255, 0)
            
            resized_masks = []
            for box, mask in zip(boxes, masks):
                xmin, ymin, xmax, ymax = box[0: 4]
                height = int(ymax - ymin)
                width = int(xmax - xmin)
                mask = resize(mask[None,...], (height, width))[0]
                mask = torch.where(mask > 20, 255, 0)
                resized_masks.append(mask)
            
            return scores, boxes, cls_idxes, rois, resized_masks
