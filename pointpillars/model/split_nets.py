import torch
import torch.nn as nn
from .quantizations import RQBottleneck
import numpy as np


class HeadNet(nn.Module):
    """Head part of PointPillars: PillarLayer + PillarEncoder"""
    def __init__(self, pillar_layer, pillar_encoder):
        super().__init__()
        self.pillar_layer = pillar_layer
        self.pillar_encoder = pillar_encoder

    def forward(self, batched_pts):
        # batched_pts: list[tensor] -> pillars, coors_batch, npoints_per_pillar
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        
        # pillars -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        
        return pillar_features

class TailNet(nn.Module):
    """Tail part of PointPillars: Backbone + Neck + Head"""
    def __init__(self, backbone, neck, head, nclasses, anchors_generator, assigners, 
                 nms_pre=100, nms_thr=0.01, score_thr=0.1, max_num=50):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.nclasses = nclasses
        self.anchors_generator = anchors_generator
        self.assigners = assigners
        self.nms_pre = nms_pre
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.max_num = max_num

    def forward(self, pillar_features, mode='test', batched_gt_bboxes=None, batched_gt_labels=None, batch_size=None):
        # pillar_features: (bs, 64, y_l, x_l)
        
        # xs: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        xs = self.backbone(pillar_features)

        # x: (bs, 384, 248, 216)
        x = self.neck(xs)

        # bbox predictions
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        
        if batch_size is None:
            batch_size = bbox_cls_pred.size(0)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            from pointpillars.model.anchors import anchor_target
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors, 
                                               batched_gt_bboxes=batched_gt_bboxes, 
                                               batched_gt_labels=batched_gt_labels, 
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        else:
            # For val and test modes
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors

def split_pointpillars(model):
    """
    Split PointPillars model into HeadNet (pillar processing) and TailNet (backbone + detection)
    Args:
        model: Original PointPillars model
    Returns:
        headnet, tailnet
    """
    headnet = HeadNet(
        pillar_layer=model.pillar_layer,
        pillar_encoder=model.pillar_encoder
    )
    
    tailnet = TailNet(
        backbone=model.backbone,
        neck=model.neck,
        head=model.head,
        nclasses=model.nclasses,
        anchors_generator=model.anchors_generator,
        assigners=model.assigners,
        nms_pre=model.nms_pre,
        nms_thr=model.nms_thr,
        score_thr=model.score_thr,
        max_num=model.max_num
    )
    
    # Copy the get_predicted_bboxes methods from original model
    tailnet.get_predicted_bboxes_single = model.get_predicted_bboxes_single
    tailnet.get_predicted_bboxes = model.get_predicted_bboxes
    
    return headnet, tailnet
