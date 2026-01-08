import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2


class CenterDORNHead(nn.Module):
    def __init__(self, model_cfg ):
        super(CenterDORNHead, self).__init__()

        self.model_cfg = model_cfg

        self.head_config = self.model_cfg.SEPARATE_HEAD_CFG

        self.head_keys = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER

        self.net_config = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT

        self.in_c = self.model_cfg.INPUT_CHANNELS

        for cur_head_name in self.head_keys:
            out_c = self.net_config[cur_head_name]['out_channels']
            conv_mid = self.net_config[cur_head_name]['conv_dim']

            fc = nn.Sequential(
                nn.Conv2d(self.in_c, conv_mid, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(conv_mid),
                nn.ReLU(),
                nn.Conv2d(conv_mid, out_c, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.__setattr__(cur_head_name, fc)

        self.forward_loss_dict=dict()

    def ordinal_regression_loss(self, depth_pred, depth_gt, valid_mask, eps=1e-8):
        """
        depth_pred: (B, D, H, W)  - predicted probability distribution over depth bins
        depth_gt:   (B, D, H, W)  - one-hot ground truth over depth bins
        valid_mask: (B, 1, H, W)  - mask of valid pixels (1 = valid, 0 = invalid)
        """

        depth_pred = torch.clamp(depth_pred, eps, 1.0 - eps)

        loss_map = -(depth_gt * torch.log(depth_pred))

        loss_map = loss_map.sum(dim=1, keepdim=True)  # (B,1,H,W)

        loss = (loss_map * valid_mask).sum() / (valid_mask.sum() + eps)

        return loss

    def get_loss(self):
        loss = 0
        for cur_name in self.head_keys:

            if cur_name == 'hm':

                pred_heatmap = self.forward_loss_dict['pred_center_dict']['hm']
                gt_heatmap = self.forward_loss_dict['gt_center_dict']['hm']

                pred = pred_heatmap.reshape(-1)
                gt = gt_heatmap.reshape(-1)

                l = F.binary_cross_entropy(torch.sigmoid(pred), gt, reduction='none')
                mask = gt > 0
                l_map = l.sum() / (mask.sum() + 1)
                # print(cur_name, l_map)
                loss+=l_map

                # print(cur_name,l_map)

            elif cur_name == 'center_dis_bin':
                pred_heatmap = self.forward_loss_dict['pred_center_dict']['center_dis_bin']
                gt_heatmap = self.forward_loss_dict['gt_center_dict']['center_dis_bin'] # B, D, W, H

                gt_res = self.forward_loss_dict['gt_center_dict']['center_dis_res']

                B, _, D, W, H = gt_res.shape

                gt_res_mask = gt_res.sum(dim=2).reshape(B,1,W,H)
                gt_res_mask = gt_res_mask>0

                gt_heatmap = gt_heatmap.reshape(B,-1,W,H)

                loss_dep = self.ordinal_regression_loss(pred_heatmap, gt_heatmap, gt_res_mask)
                loss+=loss_dep
                # print(cur_name,loss_dep)

            elif cur_name == 'center_dis_res':
                pred_h = self.forward_loss_dict['pred_center_dict'][cur_name]
                gt_h = self.forward_loss_dict['gt_center_dict'][cur_name]

                pred_h_new = pred_h.reshape(-1)
                gt_h_new = gt_h.reshape(-1)

                mask_x = torch.abs(gt_h_new) > 0

                loss_res = torch.abs(gt_h_new[mask_x] - pred_h_new[mask_x]).mean()
                # print(cur_name, loss_res)

                loss+=loss_res
                # print(cur_name,loss_res)

            else:

                pred_h = self.forward_loss_dict['pred_center_dict'][cur_name]
                gt_h = self.forward_loss_dict['gt_center_dict'][cur_name]

                pred_h_new = pred_h.reshape(-1)
                gt_h_new = gt_h.reshape(-1)

                mask_x = torch.abs(gt_h_new) > 0

                loss_res = torch.abs(gt_h_new[mask_x] - pred_h_new[mask_x]).mean()
                # print(cur_name, loss_res)
                loss+=loss_res
                # print(cur_name,loss_res)

        return loss


    def forward(self, batch_dict):

        pred_dict = {}

        gt_dict = {}

        x = batch_dict['features_2d']

        B, K, C, W, H = x.shape

        x = x.reshape(B*K, C, W, H)

        for cur_name in self.head_keys:
            pred_data = self.__getattr__(cur_name)(x)
            if cur_name == 'center_dis_bin':
                pred_data = torch.softmax(pred_data,dim=1)
            pred_dict[cur_name] = pred_data

            if self.training:
                gt_dict[cur_name] = batch_dict[cur_name]

        batch_dict['pred_center_dict'] = pred_dict

        self.forward_loss_dict['pred_center_dict'] = pred_dict

        if self.training:
            batch_dict['gt_dict'] = gt_dict
            self.forward_loss_dict['gt_center_dict'] = gt_dict

        return batch_dict
