# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from .rotated_retina_head import RotatedRetinaHead

from ..builder import ROTATED_HEADS, build_loss, build_neck
from .rotated_anchor_head import RotatedAnchorHead
from mmdet.core import images_to_levels, multi_apply, unmap
import torch
import torch.nn.functional as F
from mmrotate.core.bbox.transforms import *


@ROTATED_HEADS.register_module()
class RotatedRetinaIncrementalHeadShipRS_IECD_single40_10(RotatedRetinaHead):
    r"""An anchor-based head used in `RotatedRetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 distill=None,
                 loss_kd_cls=dict(type='KDQualityFocalLoss', beta=1, loss_weight=1.0),
                 loss_kd_bbox=dict(type='GDLoss_v1', loss_type='kld', fun='log1p', tau=1, loss_weight=1.0),
                 loss_kd_fea=dict(type='PKDLoss', loss_weight=2.0),
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.ori_num_classes = 40
        super(RotatedRetinaIncrementalHeadShipRS_psedolabel_diffclass_skysegKD_future_single40_10, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        self.distill_feat_weight = distill.get("mgd_distill_feat_weight", 0)
        self.loss_kd_cls = build_loss(loss_kd_cls)
        self.loss_kd_bbox = build_loss(loss_kd_bbox)
        self.loss_kd_fea = build_loss(loss_kd_fea)
        # self.mgd = mgdLoss()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 5.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def filter_bboxes(self, cls_scores, bbox_preds):
        """Filter predicted bounding boxes at each position of the feature
        maps. Only one bounding boxes with highest score will be left at each
        position. This filter will be used in R3Det prior to the first feature
        refinement stage.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level \
                of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]

            anchors = mlvl_anchors[lvl]

            cls_score = cls_score.permute(0, 2, 3, 1)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors,
                                          self.cls_out_channels)

            cls_score, _ = cls_score.max(dim=-1, keepdim=True)
            best_ind = cls_score.argmax(dim=-2, keepdim=True)
            best_ind = best_ind.expand(-1, -1, -1, 5)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, self.num_anchors, 5)
            best_pred = bbox_pred.gather(
                dim=-2, index=best_ind).squeeze(dim=-2)

            anchors = anchors.reshape(-1, self.num_anchors, 5)

            for img_id in range(num_imgs):
                best_ind_i = best_ind[img_id]
                best_pred_i = best_pred[img_id]
                best_anchor_i = anchors.gather(
                    dim=-2, index=best_ind_i).squeeze(dim=-2)
                best_bbox_i = self.bbox_coder.decode(best_anchor_i,
                                                     best_pred_i)
                bboxes_list[img_id].append(best_bbox_i.detach())

        return bboxes_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds):
        """This function will be used in S2ANet, whose num_anchors=1.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)

        Returns:
            list[list[Tensor]]: refined rbboxes of each level of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 5)
            anchors = mlvl_anchors[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(anchors, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list

    def forward_train(self,
                      ori_outs,
                      ori_x,
                      outs,
                      x,
                      reused_cls_scores,
                      reused_bbox_preds,
                      img_metas,
                      gt_bboxes,
                      topk_cls_inds,
                      topk_cls_scores,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      stu_features_old=None,  # 【新增】接收参数
                      tea_features_old=None,  # 【新增】接收参数
                      soft_mask_old=None, soft_mask_future=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        # outs = self(x)
        if gt_labels is None:
            loss_inputs = ori_outs + (ori_x, x) + outs + (reused_cls_scores, reused_bbox_preds) + (
            topk_cls_inds, topk_cls_scores) + (gt_bboxes, img_metas) + (stu_features_old, tea_features_old) + (
                          soft_mask_old, soft_mask_future)
        else:
            loss_inputs = ori_outs + (ori_x, x) + outs + (reused_cls_scores, reused_bbox_preds) + (
            topk_cls_inds, topk_cls_scores) + (gt_bboxes, gt_labels, img_metas) + (
                          stu_features_old, tea_features_old) + (soft_mask_old, soft_mask_future)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def loss_single(self, ori_cls_scores, ori_bbox_preds, ori_x_features, x_features, cls_score, bbox_pred, anchors,
                    labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples,
                    soft_mask_future=None):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 5).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
            weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 5).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
        """
        if soft_mask_future is not None:
            B, _, H, W = cls_score.shape
            mask_fut_resized = F.interpolate(soft_mask_future, size=(H, W), mode='bilinear',
                                             align_corners=False)  # [B, 1, H, W]
            mask_fut_resized = mask_fut_resized.squeeze(1)  # [B, H, W]

            mask_fut_anchors = mask_fut_resized.unsqueeze(-1).repeat(1, 1, 1, self.num_anchors)
            mask_fut_flat = mask_fut_anchors.reshape(-1)

            labels_flat = labels.reshape(-1)
            bg_indices = (labels_flat == self.num_classes)

            label_weights = label_weights.reshape(-1).clone()
            label_weights[bg_indices] = label_weights[bg_indices] * (1.0 - mask_fut_flat[bg_indices])

        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels.reshape(-1), label_weights, avg_factor=num_total_samples)
        # ====================================================================

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)


        return loss_cls, loss_bbox

    def distill_loss_by_image_single(self,
                                     new_cls_scores,
                                     ori_cls_inds,
                                     ori_cls_scores):
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (Tuple[int]): Stride in this scale level.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """


        new_topk_cls_scores = new_cls_scores.gather(0,
                                                    ori_cls_inds.unsqueeze(-1).expand(-1, new_cls_scores.size(-1)))
        ori_topk_cls_scores = ori_cls_scores.gather(0,
                                                    ori_cls_inds.unsqueeze(-1).expand(-1, ori_cls_scores.size(-1)))
        dist_loss_weight = 1.0
        loss_dist_cls_ERD = dist_loss_weight * self.l2_loss(new_topk_cls_scores, ori_topk_cls_scores)  # 至少看到第一次计算时为0
        return loss_dist_cls_ERD,

    @staticmethod
    def l2_loss(pred, target, reduction='mean'):
        r"""Function that takes the mean element-wise square value difference.
        """
        assert target.size() == pred.size()
        loss = (pred - target).pow(2).float()
        if reduction != 'none':
            loss = torch.mean(loss) if reduction == 'mean' else torch.sum(loss)
        return loss

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             ori_cls_scores,
             ori_bbox_preds,
             ori_x_features,
             x_features,
             cls_scores,
             bbox_preds,
             reused_cls_scores,
             reused_bbox_preds,
             topk_cls_inds,
             topk_cls_scores,
             gt_bboxes,
             gt_labels,
             img_metas,
             stu_features_old=None,  # 【新增】
             tea_features_old=None,  # 【新增】
             soft_mask_old=None, soft_mask_future=None,  # 【新增】
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            ori_cls_scores,
            ori_bbox_preds,
            ori_x_features,
            x_features,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            soft_mask_future=soft_mask_future
        )


        loss_skyseg_distill = 0.0
        n = 0

        alpha_low = 0.5
        beta_high = 2.0

        if stu_features_old is not None and tea_features_old is not None and soft_mask_old is not None:
            for s_feat, t_feat in zip(stu_features_old, tea_features_old):
                n += 1
                B, C, H, W = s_feat.shape

                mask_old_resized = F.interpolate(soft_mask_old, size=(H, W), mode='bilinear',
                                                 align_corners=False)  # [B, 1, H, W]

                s_low, s_high = self.split_freq_features(s_feat)
                t_low, t_high = self.split_freq_features(t_feat)

                loss_low_map = F.mse_loss(s_low, t_low, reduction='none')
                loss_high_map = F.mse_loss(s_high, t_high, reduction='none')

                loss_low_spatial = loss_low_map.mean(dim=1, keepdim=True)
                loss_high_spatial = loss_high_map.mean(dim=1, keepdim=True)

                weighted_loss_low = (loss_low_spatial * mask_old_resized).mean()
                weighted_loss_high = (loss_high_spatial * mask_old_resized).mean()

                loss_skyseg_distill += (weighted_loss_low + weighted_loss_high)

        if n > 0:
            loss_skyseg_distill = loss_skyseg_distill * 300
        else:
            loss_skyseg_distill = torch.tensor(0.0, requires_grad=True).to(stu_features_old[0].device)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_skyseg_distill=loss_skyseg_distill)


    def split_freq_features(self, feat, ratio=0.25):

        B, C, H, W = feat.shape

        fft_feat = torch.fft.rfft2(feat, norm='backward')

        y = torch.arange(H, device=feat.device)
        x = torch.arange(W // 2 + 1, device=feat.device)
        y = torch.min(y, H - y)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        rho = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        r = min(H, W) * ratio

        mask_low = (rho < r).float().unsqueeze(0).unsqueeze(0)
        mask_high = 1.0 - mask_low

        fft_low = fft_feat * mask_low
        fft_high = fft_feat * mask_high
        feat_low = torch.fft.irfft2(fft_low, s=(H, W), norm='backward')
        feat_high = torch.fft.irfft2(fft_high, s=(H, W), norm='backward')

        return feat_low, feat_high

