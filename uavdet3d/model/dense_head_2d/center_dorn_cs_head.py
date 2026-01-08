import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2

class SingleHead(nn.Module):
    def __init__(self, in_feat, head_keys, net_config ):
        super(SingleHead, self).__init__()

        self.head_keys = head_keys

        for cur_head_name in head_keys:
            out_c = net_config[cur_head_name]['out_channels']
            conv_mid = net_config[cur_head_name]['conv_dim']

            fc = nn.Sequential(
                nn.Conv2d(in_feat, conv_mid, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(conv_mid),
                nn.ReLU(),
                nn.Conv2d(conv_mid, out_c, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.__setattr__(cur_head_name, fc)

    def forward(self, input_x):

        pred_dict = {}

        for cur_name in self.head_keys:
            pred_data = self.__getattr__(cur_name)(input_x)
            if cur_name == 'center_dis_bin':
                pred_data = torch.softmax(pred_data,dim=1)
            pred_dict[cur_name] = pred_data

        return pred_dict


class CenterDORNCSHead(nn.Module):
    def __init__(self, model_cfg ):
        super(CenterDORNCSHead, self).__init__()

        self.model_cfg = model_cfg

        self.head_keys = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER

        self.net_config = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT

        self.cls_num = len(self.net_config)

        self.hm_config = self.model_cfg.HM

        self.in_c = self.model_cfg.INPUT_CHANNELS

        for i in range(len(self.net_config)):
            head_model = SingleHead(self.in_c, self.head_keys, self.net_config[i].DICT)

            self.__setattr__('head_'+str(i), head_model)

        self.hm_head = nn.Sequential(
                nn.Conv2d( self.in_c, self.hm_config['conv_dim'], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.hm_config['conv_dim']),
                nn.ReLU(),
                nn.Conv2d(self.hm_config['conv_dim'], self.hm_config['out_channels'], kernel_size=3, stride=1, padding=1, bias=False)
            )

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

        return loss*0.25

    def hm_loss(self, pred_heatmap, gt_heatmap):

        pred = pred_heatmap.reshape(-1)
        gt = gt_heatmap.reshape(-1)

        l = F.binary_cross_entropy(pred, gt, reduction='none')
        mask = gt > 0
        l_map = l.sum() / (mask.sum() + 1)

        return l_map


    def get_loss(self):

        loss = 0

        pred_heatmap = self.forward_loss_dict['pred_center_dict']['hm']

        gt_heatmap = self.forward_loss_dict['gt_center_dict']['hm']


        loss_hm_value = self.hm_loss(pred_heatmap, gt_heatmap)

        #print('loss_hm_value', loss_hm_value)
        loss+=loss_hm_value

        for cur_name in self.head_keys:

            if cur_name == 'center_dis_bin':
                pred_heatmap_orin = self.forward_loss_dict['pred_center_dict']['center_dis_bin']
                gt_heatmap_orin = self.forward_loss_dict['gt_center_dict']['center_dis_bin']  # B, im_num, cls, D, W, H

                B, im_num, cls, D, W, H = pred_heatmap_orin.shape

                pred_heatmap = pred_heatmap_orin.reshape(B * im_num * cls, D, W, H)
                gt_heatmap = gt_heatmap_orin.reshape(B * im_num * cls, D, W, H)

                gt_res_orin = self.forward_loss_dict['gt_center_dict']['center_dis_res']
                B, im_num, cls, D, W, H = gt_res_orin.shape
                gt_res = gt_res_orin.reshape(B * im_num * cls, D, W, H)

                gt_res_mask = gt_res.sum(dim=1).reshape(B * im_num * cls, 1, W, H)
                gt_res_mask = gt_res_mask > 0

                gt_heatmap = gt_heatmap.reshape(B * im_num * cls, -1, W, H)

                loss_dep_bin_value = self.ordinal_regression_loss(pred_heatmap, gt_heatmap, gt_res_mask)
                loss += loss_dep_bin_value
                #print(cur_name, loss_dep_bin_value)
            else:
                pred_h = self.forward_loss_dict['pred_center_dict'][cur_name]
                gt_h = self.forward_loss_dict['gt_center_dict'][cur_name]

                pred_h_new = pred_h.reshape(-1)
                gt_h_new = gt_h.reshape(-1)

                mask_x = torch.abs(gt_h_new) > 0

                loss_res_value = torch.abs(gt_h_new[mask_x] - pred_h_new[mask_x]).mean()
                # print(cur_name, loss_res)
                loss+=loss_res_value
                #print(cur_name, loss_res_value)

        return loss

    def forward(self, batch_dict):

        pred_dict = {}

        gt_dict = {}

        x = batch_dict['features_2d']

        B, im_num, C, W, H = x.shape

        x = x.reshape(B*im_num, C, W, H) # B, im, C, W, H

        for key in self.head_keys:
            pred_dict[key]=[]
            if self.training:
                gt_dict[key] = batch_dict[key]

        for i in range(self.cls_num):

            pred_dict_single_head = self.__getattr__('head_'+str(i))(x)

            for k in pred_dict_single_head:

                pred_data = pred_dict_single_head[k]

                if k == 'center_dis_bin':
                    pred_data = torch.softmax(pred_data, dim=1)

                B,C,W,H = pred_data.shape

                pred_dict[k].append(pred_dict_single_head[k].reshape(B, im_num, 1, C, W, H)) # B, im, cls, C, W, H

        for key in self.head_keys:
            pred_dict[key] = torch.concatenate(pred_dict[key], dim=2)

        pred_hm = self.hm_head(x)
        B, C, W, H = pred_hm.shape
        pred_hm = pred_hm.reshape(B, im_num, C, W, H)

        pred_hm = torch.sigmoid(pred_hm)

        pred_dict['hm'] = pred_hm

        for cur_name in self.head_keys:

            if self.training:
                gt_dict[cur_name] = batch_dict[cur_name]

        batch_dict['pred_center_dict'] = pred_dict

        self.forward_loss_dict['pred_center_dict'] = pred_dict

        if self.training:
            gt_dict['hm'] = batch_dict['hm']
            batch_dict['gt_dict'] = gt_dict
            self.forward_loss_dict['gt_center_dict'] = gt_dict

        return batch_dict
