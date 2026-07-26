# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector
from .rotated_retinanet import RotatedRetinaNet

import torch
import os
from collections import OrderedDict
from mmengine.runner.checkpoint import load_checkpoint, load_state_dict
from mmengine import Config
from mmrotate.models import build_detector
import copy
from mmdet.core import images_to_levels, multi_apply, unmap
import torch.nn.functional as F

from ...core.bbox.iou_calculators.rotate_iou2d_calculator import RBboxOverlaps2D
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmrotate  # noqa: F401
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import math
import glob
import os.path as osp
import numpy as np

# 引入 MMRotate 的工具
import cv2
from mmrotate.core import obb2poly
from SkySense_O import build_model_SkySenseO   # You need to implement it yourself, according to https://github.com/zqcrafts/SkySense-O/blob/main/demo/readme.md

from mmrotate.datasets.builder import PIPELINES
from mmcv.utils import build_from_cfg
from mmdet.datasets.pipelines import Compose  # Compose 通常复用 MMDet 的


@ROTATED_DETECTORS.register_module()
class RotatedRetinaNetIncrementalShipRS_IECD_single40_10(RotatedRetinaNet):
    """Implementation of Rotated `RetinaNet.`__

    __ https://arxiv.org/abs/1708.02002
    """

    def __init__(self,
                 ori_setting,  # +
                 current_dataset_setting,  # +
                 pseudo_label_setting,  # +
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 dist_loss_weight=1,  # +
                 latest_model_flag=True):  # +
        super(RotatedRetinaNetIncrementalShipRS_psedolabel_diffclass_skysegKD_future_single40_10,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)

        self.dist_loss_weight = dist_loss_weight
        self.reused_teacher_head_idx = 3
        if latest_model_flag:
            self.load_base_detector(ori_setting)
            self._is_init = True

        self.ori_num_classes = ori_setting['ori_num_classes']
        self.use_weight_pseudo_label = pseudo_label_setting['is_use']

        if self.use_weight_pseudo_label:
            self.pseudo_label_alpha = pseudo_label_setting['alpha']
            if current_dataset_setting['dataset'] == 'ShipRS':
                self.initial_gauss_mixture_model_ShipRS(current_dataset_setting, ori_setting)

        self.SHIP_CLASS_INDEX = 184
        self.BOAT_CLASS_INDEX = 37
        self.airplane_CLASS_INDEX = 111
        self.aircraft_CLASS_INDEX = 161
        self.vehicle_CLASS_INDEX = 6
        self.car_CLASS_INDEX = 10
        self.truck_CLASS_INDEX = 14
        self.heavy_truck_CLASS_INDEX = 209
        self.van_CLASS_INDEX = 74
        self.bus_CLASS_INDEX = 76
        self.coach_bus_CLASS_INDEX = 82
        self.trailer_CLASS_INDEX = 135
        self.tennis_court_CLASS_INDEX = 28
        self.basketball_court_CLASS_INDEX = 38
        self.baseball_field_CLASS_INDEX = 81
        self.soccer_field_CLASS_INDEX = 88
        self.sport_field_CLASS_INDEX = 159
        self.road_CLASS_INDEX = 7
        self.roundabout_CLASS_INDEX = 35
        self.bridge_CLASS_INDEX = 57
        self.SHIP_CLASS_INDEX = 184
        self.BOAT_CLASS_INDEX = 37
        self.SkySenseO = build_model_SkySenseO()
        self.SkySenseO.eval()
        print("SkySenseO model built successfully.")
        self.clean_pipeline = self._build_clean_pipeline()

    def _build_clean_pipeline(self):
        pipeline_cfg = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ]

        transforms = []
        for transform_cfg in pipeline_cfg:
            transform = build_from_cfg(transform_cfg, PIPELINES)
            transforms.append(transform)

        return Compose(transforms)

    def initial_gauss_mixture_model_ShipRS(self, current_dataset_setting, ori_setting, model_type='GMM'):
        class_names = ['Test-Ship', 'Ticonderoga', 'Training-Ship', 'Tugboat', 'Wasp-LL',
                       'Yacht', 'YuDao-LL', 'YuDeng-LL', 'YuTing-LL', 'YuZhao-LL',
                       'AOE', 'Arleigh-Burke-DD', 'Asagiri-DD', 'Atago-DD', 'Austin-LL', 'Barge', 'Cargo',
                       'Commander', 'Container-Ship', 'Dock', 'EPF', 'Enterprise', 'Ferry', 'Fishing-Vessel',
                       'Hatsuyuki-DD', 'Hovercraft',
                       'Hyuga-DD', 'LHA-LL', 'LSD-41-LL', 'Masyuu-AS', 'Medical-Ship', 'Midway', 'Motorboat', 'Nimitz',
                       'Oil-Tanker',
                       'Osumi-LL', 'Other-Aircraft-Carrier', 'Other-Auxiliary-Ship', 'Other-Destroyer', 'Other-Frigate',
                       'Other-Landing',
                       'Other-Merchant', 'Other-Ship', 'Other-Warship', 'Patrol', 'Perry-FF', 'RoRo', 'Sailboat',
                       'Sanantonio-AS', 'Submarine']
        see_class = class_names[0: self.bbox_head.num_classes - self.ori_num_classes]
        old_class = class_names[self.bbox_head.num_classes - self.ori_num_classes: self.bbox_head.num_classes]
        cls_mapp = {c: i
                    for i, c in enumerate(see_class)
                    }
        cls_mapp_fanzhuan = {i: c
                             for i, c in enumerate(see_class)
                             }
        if current_dataset_setting['cache_path'] is not None and os.path.exists(current_dataset_setting['cache_path']):
            print("use confidence cache from: {}".format(current_dataset_setting['cache_path']))
            labels_confidences = torch.load(current_dataset_setting['cache_path'])
        else:
            cfg_path = ori_setting['ori_config_file']
            pt_path = ori_setting['ori_checkpoint_file']
            model = init_detector(cfg_path, pt_path)
            ann_dir = current_dataset_setting['ann_dir']
            img_dir = current_dataset_setting['img_dir']
            labels_confidences = []
            iou_calculator = RBboxOverlaps2D()

            for i in range(ori_setting['ori_num_classes']):
                labels_confidences.append([])

            ann_files = glob.glob(ann_dir + '/*.txt')
            for ann_file in tqdm(ann_files):
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                ann_name = img_id + '.txt'
                img_file = os.path.join(img_dir, img_name)
                xml_file = os.path.join(ann_dir, ann_name)
                gt_bboxes = []
                gt_labels = []


                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, 'le90')
                        except:
                            continue
                        cls_name = bbox_info[8]
                        label_ = cls_mapp[cls_name]
                        gt_bboxes.append([x, y, w, h, a])
                        gt_labels.append(label_)

                result = inference_detector(model, img_file)
                labels, bboxes, scores = self.parse_detection_result(result, self.ori_num_classes)

                for label, score, bbox in zip(labels, scores, bboxes):
                    skip_tag = False
                    for g_bbox, g_label in zip(gt_bboxes, gt_labels):
                        if cls_mapp_fanzhuan[g_label] in see_class:
                            iou = iou_calculator(bbox.unsqueeze(0), torch.tensor(g_bbox).unsqueeze(0))
                            if iou > 0.8:
                                skip_tag = True
                    if skip_tag:
                        continue
                    label = int(label)
                    score = float(score)
                    labels_confidences[label].append(score)

            if current_dataset_setting['cache_path'] is not None:
                torch.save(labels_confidences, current_dataset_setting['cache_path'])

        self.negative_threshold = torch.Tensor([])
        self.positive_threshold = torch.Tensor([])


        gmm_means = []
        gmm_covs = []
        gmm_weights = []

        self.gauss_model = GaussianMixture(n_components=2, max_iter=5000, random_state=1234, covariance_type='tied')
        for confidence in labels_confidences:
            if len(confidence) < 2:
                confidence = [0.0, 1.0]

            train_data = torch.Tensor(confidence).reshape(-1, 1)
            self.gauss_model.fit(train_data)
            means = self.gauss_model.means_
            covs = self.gauss_model.covariances_
            weights = self.gauss_model.weights_

            if means[0][0] > means[1][0]:
                means = np.array([means[1], means[0]])
                weights = np.array([weights[1], weights[0]])

            gmm_means.append(means)
            gmm_covs.append(covs)
            gmm_weights.append(weights)


            self.negative_threshold = torch.cat([self.negative_threshold, torch.Tensor([(means[1] + means[0]) / 2.])],
                                                dim=0)
            self.positive_threshold = torch.cat([self.positive_threshold, torch.Tensor(
                [means[1][0] - self.pseudo_label_alpha * (math.sqrt(covs[0][0]))])], dim=0)

        self.register_buffer('gmm_means',
                             torch.tensor(gmm_means, dtype=torch.float32).squeeze(-1))
        self.register_buffer('gmm_covs',
                             torch.tensor(gmm_covs, dtype=torch.float32).squeeze(-1).squeeze(-1))
        self.register_buffer('gmm_weights', torch.tensor(gmm_weights, dtype=torch.float32))

        print("GMM negative_threshold: {}, positive_threshold: {}".format(self.negative_threshold,
                                                                          self.positive_threshold))
        print("GMM params registered for Soft Mask generation.")

    def parse_detection_result(self, result, num_classes):
        if len(result) != num_classes:
            raise ValueError(f"预期 {num_classes} 个类别，但得到 {len(result)}")

        labels_list = []
        bboxes_list = []
        scores_list = []

        for cls_id in range(num_classes):
            cls_array = result[cls_id]  # (N, 6)
            if cls_array.size == 0:
                continue
            num_dets = cls_array.shape[0]
            labels_list.extend([cls_id] * num_dets)
            bboxes_list.append(cls_array[:, :5])  # (N, 5)
            scores_list.append(cls_array[:, 5])  # (N,)

        if labels_list:
            labels = np.array(labels_list)
            bboxes = np.concatenate(bboxes_list, axis=0)
            scores = np.concatenate(scores_list, axis=0)
        else:
            labels = np.empty(0, dtype=np.int64)
            bboxes = np.empty((0, 5), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)

        labels = torch.from_numpy(labels).to(torch.int64)
        bboxes = torch.from_numpy(bboxes).to(torch.float32)
        scores = torch.from_numpy(scores).to(torch.float32)

        return labels, bboxes, scores

    def _load_checkpoint_for_new_model(self, checkpoint_file, map_location=None, strict=False, logger=None):
        # load ckpt
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(checkpoint_file))
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
                                       v in checkpoint['state_dict'].items()}

        orig_cls_weight = state_dict['bbox_head.retina_cls.weight']  # shape: [360, 256, 3, 3] (9 * 40)
        orig_cls_bias = state_dict['bbox_head.retina_cls.bias']  # shape: [360]

        new_cls_weight = self.bbox_head.retina_cls.weight.detach().clone()  # shape: [450, 256, 3, 3] (9 * 50)
        new_cls_bias = self.bbox_head.retina_cls.bias.detach().clone()  # shape: [450]


        for i in range(9):
            start_idx = i * 40
            end_idx = (i + 1) * 40
            for j in range(40):
                new_cls_weight[i * 50 + j + 10, :, :, :] = orig_cls_weight[start_idx + j, :, :, :]
            new_cls_bias[i * 50 + 10: i * 50 + 10 + 40] = orig_cls_bias[start_idx:end_idx]

        state_dict['bbox_head.retina_cls.weight'] = new_cls_weight
        state_dict['bbox_head.retina_cls.bias'] = new_cls_bias

        # load state_dict
        if hasattr(self, 'module'):
            load_state_dict(self.module, state_dict, strict, logger)
        else:
            load_state_dict(self, state_dict, strict, logger)

    def load_base_detector(self, ori_setting):
        """
                Initialize detector from config file.
        :param ori_setting:
        :return:
        """
        assert os.path.isfile(ori_setting['ori_checkpoint_file']), '{} is not a valid file'.format(
            ori_setting['ori_checkpoint_file'])
        ##### init original model & frozen it ！！！！！！！！！！！！！！！！！！！！！！！！！！ #####
        # build model
        ori_cfg = Config.fromfile(ori_setting['ori_config_file'])
        if hasattr(ori_cfg.model, 'latest_model_flag'):
            ori_cfg.model.latest_model_flag = False
        # ori_model = MODELS.build(ori_cfg.model)
        ori_model = build_detector(
            ori_cfg.model,
            train_cfg=ori_cfg.get('train_cfg'),
            test_cfg=ori_cfg.get('test_cfg'))
        # load checkpoint
        load_checkpoint(ori_model, ori_setting.ori_checkpoint_file, strict=True)
        # # set to eval mode
        ori_model.eval()
        # ori_model.forward = ori_model.forward_dummy
        # # set requires_grad of all parameters to False
        for param in ori_model.parameters():
            param.requires_grad = False

        # ##### init original branchs of new model #####
        self.ori_num_classes = ori_setting.ori_num_classes
        self._load_checkpoint_for_new_model(ori_setting.ori_checkpoint_file)
        print('======> load base checkpoint for new model from {}'.format(ori_setting.ori_checkpoint_file))
        self.ori_model = ori_model

    def generate_old_class_mask(self, img_tensor, img_metas):
        B, C, _, _ = img_tensor.shape
        device = img_tensor.device
        final_masks = []

        TARGET_SIZE = 1024

        for i in range(B):

            current_img_path = img_metas[i]['filename']
            ori_h, ori_w = img_metas[i]['ori_shape'][:2]

            with torch.no_grad():
                seg_result = self.SkySenseO.sam_seg(current_img_path, "open_world")


            mask_ship = ((seg_result == self.SHIP_CLASS_INDEX) | (
                        seg_result == self.BOAT_CLASS_INDEX)).cpu().numpy().astype(np.uint8)


            gtbbox_path = current_img_path.replace('imgpng', 'labelTxt').rsplit('.', 1)[
                              0] + '.txt'

            mask_gt_overlap = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

            if os.path.exists(gtbbox_path):
                with open(gtbbox_path, 'r') as f:
                    lines = f.readlines()

                polys_to_draw = []

                scale_x = TARGET_SIZE / ori_w
                scale_y = TARGET_SIZE / ori_h

                for line in lines:

                    parts = line.strip().split()
                    if len(parts) < 9: continue

                    coords = list(map(float, parts[:8]))  # [x1, y1, ..., y4]

                    coords[0::2] = [x * scale_x for x in coords[0::2]]
                    coords[1::2] = [y * scale_y for y in coords[1::2]]

                    poly = np.array(coords).reshape(4, 2).astype(np.int32)
                    polys_to_draw.append(poly)

                if polys_to_draw:
                    cv2.fillPoly(mask_gt_overlap, polys_to_draw, 1)

            mask_final_np = mask_ship * (1 - mask_gt_overlap)

            mask_final = torch.from_numpy(mask_final_np).to(device).float()
            final_masks.append(mask_final)

        return torch.stack(final_masks).unsqueeze(1)

    def calculate_gmm_posterior(self, scores, class_indices):
        means = self.gmm_means[class_indices]
        covs = self.gmm_covs[class_indices]
        weights = self.gmm_weights[class_indices]


        var = covs + 1e-6
        norm_const = 1.0 / torch.sqrt(2 * math.pi * var)

        pdf_0 = norm_const * torch.exp(-((scores - means[..., 0]) ** 2) / (2 * var))
        pdf_1 = norm_const * torch.exp(-((scores - means[..., 1]) ** 2) / (2 * var))

        prob_0 = weights[..., 0] * pdf_0
        prob_1 = weights[..., 1] * pdf_1

        p_old = prob_1 / (prob_0 + prob_1 + 1e-8)
        return p_old

    def sel_pos_single(self, cat_cls_scores):

        cat_conf = cat_cls_scores.sigmoid()
        max_scores, _ = cat_conf.max(dim=-1)

        cls_thr = max_scores.mean() + 2 * max_scores.std()
        valid_mask = max_scores > cls_thr
        topk_cls_inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        topk_cls_scores = cat_cls_scores.gather(
            0, topk_cls_inds.unsqueeze(-1).expand(-1, cat_cls_scores.size(-1)))

        return topk_cls_inds, topk_cls_scores

    def sel_pos(self, cls_scores, bbox_preds):
        """Select positive predictions based on classification scores.

        Args:
            model (nn.Module): The loaded detector.
            cls_scores (List[Tensor]): Classification scores for each FPN level.
            bbox_preds (List[Tensor]): BBox predictions for each FPN level.
            #centernesses (List[Tensor]): Centernesses predictions for each FPN level.

        Returns:
            cat_cls_scores (Tensor): FPN concatenated classification scores.
            #cat_centernesses (Tensor): FPN concatenated centernesses.
            topk_bbox_preds (Tensor): Selected top-k bbox predictions.
            topk_inds (Tensor): Selected top-k indices.
        """
        assert len(cls_scores) == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)
        cat_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.ori_model.bbox_head.cls_out_channels)
            for cls_score in cls_scores
        ]

        cat_cls_scores = torch.cat(cat_cls_scores, dim=1)  # 将所有尺度预测结果合并 [2,20267,40]

        topk_cls_inds, topk_cls_scores = multi_apply(
            self.sel_pos_single,
            cat_cls_scores)

        return topk_cls_inds, topk_cls_scores

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)

        clean_imgs_list = []
        for i in range(len(img_metas)):

            results = {
                'img_prefix': None,
                'img_info': {'filename': img_metas[i]['filename']},

                'flip': False,
                'flip_direction': None,
            }

            processed_data = self.clean_pipeline(results)

            clean_tensor = processed_data['img'].data
            clean_imgs_list.append(clean_tensor)

        clean_imgs = torch.stack(clean_imgs_list).to(
            img.device)

        mask_skysense_binary = self.generate_old_class_mask(clean_imgs, img_metas)

        with torch.no_grad():
            tea_features_old = self.ori_model.extract_feat(clean_imgs)
            tea_cls_scores_clean = self.ori_model.bbox_head(tea_features_old)[0]

        tea_cls_p3 = tea_cls_scores_clean[0]
        B, C, H_p3, W_p3 = tea_cls_p3.shape
        num_anchors = 9
        ori_num_classes = self.ori_num_classes

        tea_cls_p3 = tea_cls_p3.view(B, num_anchors, ori_num_classes, H_p3, W_p3)
        tea_cls_prob = tea_cls_p3.sigmoid()

        max_scores_anchors, _ = tea_cls_prob.max(dim=1)

        max_scores, max_classes = max_scores_anchors.max(dim=1)


        p_old_p3 = self.calculate_gmm_posterior(max_scores, max_classes)
        p_old_p3 = p_old_p3.unsqueeze(1)


        p_old_full = F.interpolate(p_old_p3, size=(clean_imgs.shape[2], clean_imgs.shape[3]), mode='bilinear',
                                   align_corners=False)


        soft_mask_old = mask_skysense_binary * p_old_full
        soft_mask_future = mask_skysense_binary * (1.0 - p_old_full)

        stu_features_old = self.extract_feat(clean_imgs)
        x = self.extract_feat(img)


        ori_x = self.ori_model.extract_feat(img)

        # psedolabel
        psedolabel_outs = self.ori_model.bbox_head(ori_x)
        psedolabel = self.ori_model.bbox_head.get_bboxes(*psedolabel_outs, img_metas)
        psedolabel_bs = len(psedolabel)
        iou_calculator = RBboxOverlaps2D()
        for img_id in range(psedolabel_bs):
            bbox = torch.tensor([])
            score = torch.tensor([])
            label = torch.tensor([])
            keep = torch.tensor([])
            pse_label = psedolabel[img_id]


            if len(pse_label) == 0 or len(pse_label[0]) == 0:
                continue

            bbox = pse_label[0][:, :5]
            score = pse_label[0][:, 5]
            label = pse_label[1]
            self.positive_threshold = self.positive_threshold.to(score.device)
            keep = torch.tensor(
                [score[i] > self.positive_threshold[label[i]] for i in range(len(label))])  # 不同类别的过滤阈值不同
            bbox = bbox[keep]
            label = label[keep]

            if len(bbox) != len(label):
                continue
            label = label + (self.bbox_head.num_classes - self.ori_num_classes)

            if len(label) != 0 or len(bbox) != 0:
                keep2 = torch.ones(len(bbox), dtype=torch.bool)
                i = 0
                for pse_bbox in bbox:
                    for gt_bbox in gt_bboxes[img_id]:
                        iou = iou_calculator(gt_bbox.unsqueeze(0), pse_bbox.unsqueeze(0))
                        if iou > 0.7:
                            keep2[i] = 0
                    i = i + 1
                bbox = bbox[keep2]
                label = label[keep2]

            gt_bboxes[img_id] = torch.cat((gt_bboxes[img_id], bbox), dim=0)
            gt_labels[img_id] = torch.cat((gt_labels[img_id], label), dim=0)


        tea_cls_scores, tea_bbox_preds, tea_cls_hold, tea_reg_hold = \
            multi_apply(self.forward_crosskd_single, ori_x, module=self.ori_model)
        ori_outs = (tea_cls_scores, tea_bbox_preds)

        topk_cls_inds, topk_cls_scores = self.sel_pos(*ori_outs)

        x = self.extract_feat(img)
        stu_cls_scores, stu_bbox_preds, stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_crosskd_single, x, module=self)
        outs = (stu_cls_scores, stu_bbox_preds)

        reused_cls_scores, reused_bbox_preds = multi_apply(
            self.reuse_teacher_head, tea_cls_hold, tea_reg_hold, stu_cls_hold,
            stu_reg_hold)
        reused_cls_scores = tuple(reused_cls_scores)
        reused_bbox_preds = tuple(reused_bbox_preds)

        losses = self.bbox_head.forward_train(ori_outs, ori_x, outs, x,
                                              reused_cls_scores, reused_bbox_preds,
                                              img_metas, gt_bboxes,
                                              topk_cls_inds,
                                              topk_cls_scores,
                                              gt_labels,
                                              gt_bboxes_ignore,
                                              stu_features_old=stu_features_old,
                                              tea_features_old=tea_features_old,
                                              soft_mask_old=soft_mask_old,
                                              soft_mask_future=soft_mask_future
                                              )
        return losses
