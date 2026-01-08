from uavdet3d.datasets import DatasetTemplate
from uavdet3d.datasets.mav6d.eval import eval_6d_pose
import numpy as np
import os
import torch
import pandas as pd
import cv2
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from uavdet3d.datasets.mav6d.mav6d_utils import draw_box9d_on_image
import copy

#matplotlib.use('tkAGG')

class CARLA_Det_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training, root_path, logger):
        super(CARLA_Det_Dataset, self).__init__(dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger)

        self.dataset_cfg=dataset_cfg
        self.root_path=root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.training=training
        self.logger=logger

        self.im_path_name = 'images'
        self.label_path_name = 'boxes'

        self.intrinsic = self.get_camera_intrinsic_matrix()
        self.distortion_matrix = np.array([0, 0, 0, 0, 0])
        self.extrinsic = np.eye(4)

        self.raw_im_width = dataset_cfg.IM_SIZE[0]
        self.raw_im_hight = dataset_cfg.IM_SIZE[1]

        self.new_im_width = dataset_cfg.IM_RESIZE[0]
        self.new_im_hight = dataset_cfg.IM_RESIZE[1]

        self.im_num = self.dataset_cfg.IM_NUM

        self.obj_size = np.array(self.dataset_cfg.OB_SIZE)

        self.center_rad = self.dataset_cfg.CENTER_RAD

        self.stride = self.dataset_cfg.STRIDE

        self.class_name_dict = self.dataset_cfg.CLASS_NAMES


        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]


        self.sample_scene_list = []

        split_dir = os.path.join(str(self.root_path) , 'split', self.split + '.txt')
        self.seq_list = [x.strip() for x in open(split_dir).readlines()]

        for seq_name in self.seq_list:

            seq_path = os.path.join(str(self.root_path), 'raw_data', seq_name, 'images')

            all_frames = os.listdir(seq_path)

            for frame in all_frames:

                path_info = [seq_name, frame]

                self.sample_scene_list.append(path_info)


        self.infos = []
        self.include_CARLA_data(self.mode)

    def get_camera_intrinsic_matrix(self, image_width=1920, image_height=1080, fov=90):
        fx = image_width / (2.0 * np.tan(fov * np.pi / 360.0))
        fy = fx  # Assuming square pixels
        cx = image_width / 2.0
        cy = image_height / 2.0
        intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return intrinsic_matrix

    def set_split(self, split):
        super(CARLA_Det_Dataset, self).__init__(dataset_cfg=self.dataset_cfg, training=self.training,
                                                 root_path=self.root_path, logger=self.logger)

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.sample_scene_list = []

        split_dir = os.path.join(self.root_path, 'split', self.split + '.txt')
        self.seq_list = [x.strip() for x in open(split_dir).readlines()]

        for seq_name in self.seq_list:

            seq_path = os.path.join(self.root_path, 'raw_data', seq_name, 'images')

            all_frames = os.listdir(seq_path)

            for frame in all_frames:
                path_info = [seq_name, frame]

                self.sample_scene_list.append(path_info)

        self.infos = []
        self.include_CARLA_data(self.mode)

    def include_CARLA_data(self, mode):

        self.logger.info('Loading CARLA dataset')

        CARLA_infos = []

        root_path = str(self.root_path)

        for i in range(0, len(self.sample_scene_list), self.dataset_cfg.SAMPLED_INTERVAL[mode]):

            info_item = self.sample_scene_list[i]

            seq_name = info_item[0]

            frame_name = info_item[1]

            each_im_path = os.path.join(root_path, 'raw_data', seq_name, self.im_path_name, frame_name)
            each_label_path = os.path.join(root_path, 'raw_data', seq_name,  self.label_path_name, frame_name.replace('png', 'pkl'))

            data_info = {'im_path': each_im_path, 'label_path': each_label_path, 'seq_id': seq_name, 'frame_id': frame_name, 'cls_name': self.class_name_dict}


            CARLA_infos.append(data_info)

        self.infos = CARLA_infos


    def __len__(self):

        return len(self.infos)

    def __getitem__(self, item):

        each_info = self.infos[item]

        im_path = each_info['im_path']
        cls_name = each_info['cls_name']
        label_path = each_info['label_path']
        seq_id = each_info['seq_id']
        frame_id = each_info['frame_id']

        image = cv2.imread(im_path).astype(np.float32)
        resized_img = cv2.resize(
            image,
            (self.new_im_width, self.new_im_hight),  # 目标尺寸 (width, height)
            interpolation=cv2.INTER_AREA  # 缩小推荐使用区域插值
        )
        resized_img = resized_img.transpose(2,0,1)
        C,W,H = resized_img.shape
        resized_img = resized_img.reshape(1,C,W,H)

        with open(label_path, 'rb') as f:
            boxes9d = pickle.load(f)

        boxes9d = np.array(boxes9d)

        data_dict = {}

        data_dict['intrinsic'] = np.array([self.intrinsic])
        data_dict['extrinsic'] = np.array([self.extrinsic])
        data_dict['distortion'] = np.array([self.distortion_matrix])
        data_dict['raw_im_size'] = np.array([self.raw_im_width, self.raw_im_hight])
        data_dict['new_im_size'] = np.array([self.new_im_width, self.new_im_hight])
        data_dict['obj_size'] = np.array(self.dataset_cfg.OB_SIZE)
        data_dict['seq_id'] = seq_id
        data_dict['frame_id'] = frame_id
        data_dict['stride'] = self.stride
        data_dict['image'] = resized_img
        data_dict['gt_box9d'] = boxes9d
        data_dict['gt_name'] = np.array([cls_name[0]]*len(boxes9d))

        data_dict = self.data_pre_processor(data_dict)

        return data_dict


    def name_from_code(self, name_indices):

        names = [self.class_name_dict[int(x)] for x in name_indices]

        return np.array(names)


    def generate_prediction_dicts(self, batch_dict, output_path):

        batch_size = batch_dict['batch_size']

        annos = []

        for batch_id in range(batch_size):

            pred_boxes9d = batch_dict['pred_boxes9d'][batch_id] # 1, 1, 4, W, H

            seq_id = batch_dict['seq_id'][batch_id]
            frame_id = batch_dict['frame_id'][batch_id]

            intrinsic = batch_dict['intrinsic'][batch_id] # 3, 3
            extrinsic = batch_dict['extrinsic'][batch_id] # 4, 4
            distortion = batch_dict['distortion'][batch_id] # 5,
            raw_im_size = batch_dict['raw_im_size'][batch_id] # 2,
            new_im_size = batch_dict['new_im_size'][batch_id] # 2,
            obj_size = batch_dict['obj_size'][batch_id] # 3,

            # key_points_2d = batch_dict['key_points_2d'][batch_id]
            confidence = batch_dict['confidence'][batch_id]
            im_path = os.path.join(self.root_path, 'raw_data', seq_id, self.im_path_name,frame_id)

            pred_name = self.name_from_code(pred_boxes9d[:,-1])

            frame_dict = {'seq_id': seq_id,
                          'frame_id': frame_id,
                          'intrinsic': intrinsic,
                          'extrinsic': extrinsic,
                          'distortion': distortion,
                          'raw_im_size': raw_im_size,
                          'obj_size': obj_size,
                          #'key_points_2d': key_points_2d,
                          'confidence': confidence,
                          'pred_box9d': pred_boxes9d,
                          'pred_name': pred_name,
                          'im_path': im_path
                          }

            if 'gt_box9d' in batch_dict:
                gt_box9d = batch_dict['gt_box9d'][batch_id].cpu().numpy() # 1, 9
                #gt_pts2d = batch_dict['gt_pts2d'][batch_id].cpu().numpy() # 1, 2
                frame_dict['gt_box9d'] = gt_box9d
                gt_name = batch_dict['gt_name'][batch_id]
                frame_dict['gt_name'] = gt_name

                # frame_dict['gt_pts2d']=gt_pts2d

            annos.append(frame_dict)

            image = cv2.imread(im_path)
            #
            # image = draw_2d_points_on_image(key_points_2d,
            #                                 image,
            #                                 color=(0,255,0),
            #                                 radius=8)

            image = draw_box9d_on_image(pred_boxes9d, image,
                                        img_width=raw_im_size[0],
                                        img_height=raw_im_size[1],
                                        color=(255, 0, 0),
                                        intrinsic_mat=intrinsic[0],
                                        extrinsic_mat=extrinsic[0],
                                        distortion_matrix=distortion[0],
                                        offset=np.array([0., 0., 0.]))



            image = draw_box9d_on_image(gt_box9d, image,
                                        img_width=raw_im_size[0],
                                        img_height=raw_im_size[1],
                                        color=(0, 0, 255),
                                        intrinsic_mat=intrinsic[0],
                                        extrinsic_mat=extrinsic[0],
                                        distortion_matrix=distortion[0],
                                        offset=np.array([0., 0., 0.]))

            cv2.imshow('im', image)
            cv2.waitKey(1)

        return annos

    def evaluation_by_name(self, annos, metric_root_path, cls_name):

        orin_error, pos_error = eval_6d_pose(annos, max_dis=self.dataset_cfg.MAX_DIS)

        plt.scatter(np.arange(0,len(pos_error)), pos_error)
        plt.savefig(os.path.join(metric_root_path, cls_name+'_pos_error.png'))

        print(cls_name+'_orin_error： ', orin_error.mean())
        print(cls_name+'_pose_error: ', pos_error.mean())

        print(cls_name+'_orin_error_median: ', np.median(orin_error))
        print(cls_name+'_pose_error_median: ', np.median(pos_error))

        print(cls_name+'_orin_error_min： ', orin_error.min())
        print(cls_name+'_pose_error_min: ', pos_error.min())

        print(cls_name+'_orin_error_max： ', orin_error.max())
        print(cls_name+'_pose_error_max: ', pos_error.max())

        return_str = '\n'+cls_name+': \n orin_error： ' + str(orin_error.mean()) + '\n'\
                     + 'pose_error: ' + str(pos_error.mean()) + '\n'\
                     + 'orin_error_median: ' + str(np.median(orin_error)) +'\n' \
                     + 'pos_error_median: ' + str(np.median(pos_error)) + '\n' \
                     + 'orin_error_min: ' + str(orin_error.min()) + '\n' \
                     + 'pos_error_min: ' + str(pos_error.min()) + '\n' \
                     + 'orin_error_max: ' + str(orin_error.max()) + '\n' \
                     + 'pose_error_max: ' + str(pos_error.max()) + '\n'

        orin_error_out_path = os.path.join(str(metric_root_path),cls_name+'_orin_error.csv')
        orin_error_pd = pd.DataFrame({cls_name+'_orin_error': orin_error.tolist()})
        orin_error_pd.to_csv(orin_error_out_path)

        pos_error_out_path = os.path.join(str(metric_root_path),cls_name+'_pos_error.csv')
        pos_error_pd = pd.DataFrame({cls_name+'_pos_error': pos_error.tolist()})
        pos_error_pd.to_csv(pos_error_out_path)

        mean_error_out_path = os.path.join(str(metric_root_path), cls_name+'_mean_error.csv')
        mean_error_pd = pd.DataFrame({ cls_name+'_orin_error': [orin_error.mean()], cls_name+'_pose_error': [pos_error.mean()] })
        mean_error_pd.to_csv(mean_error_out_path)

        return return_str


    def evaluation(self, annos, metric_root_path):

        all_names = self.class_name_dict

        final_str = ''

        for cls_n in all_names:

            this_annos_pred = copy.deepcopy(annos)

            new_annos = []

            for each_anno in this_annos_pred:

                new_anno = {}
                gt_name = each_anno['gt_name']
                gt_box = each_anno['gt_box9d']
                pred_name = each_anno['pred_name']
                pre_box = each_anno['pred_box9d']

                k = gt_box.__len__()-1

                while k>0 and gt_box[k].sum()==0:
                    k-=1

                gt_box = gt_box[:k+1]

                name_mask = gt_name==cls_n
                new_anno['gt_box9d'] = gt_box[name_mask]

                name_mask = pred_name==cls_n

                new_anno['pred_box9d'] = pre_box[name_mask]

                if new_anno['gt_box9d'].shape[0]>=1:

                    new_annos.append(new_anno)

            if len(new_annos)>=2:
                this_str = self.evaluation_by_name(new_annos, metric_root_path, cls_name = cls_n)

                final_str+=this_str

        this_str = self.evaluation_by_name(annos, metric_root_path, cls_name='all')

        final_str += this_str

        return final_str



