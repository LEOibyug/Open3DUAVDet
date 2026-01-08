from uavdet3d.datasets import DatasetTemplate
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
import pickle as pkl
from .ads_metric import LAA3D_ADS_Metric
from scipy.spatial.transform import Rotation as R

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

#matplotlib.use('tkAGG')


def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) between two 2D bounding boxes.
    Boxes are in the format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

def match_gt(gt_box9d, pred_boxes9d, gt_box2d, pred_box2d, match_iou=0.1):
    """
    Match predicted 3D boxes to ground-truth 3D boxes based on 2D IoU.

    Args:
        gt_box9d: [M, 9] Ground-truth 3D boxes
        pred_boxes9d: [N, 9] Predicted 3D boxes
        gt_box2d: [M, 4] Ground-truth 2D boxes [x1, y1, x2, y2]
        pred_box2d: [N, 4] Predicted 2D boxes
        match_iou: Minimum IoU threshold for a valid match

    Returns:
        matched_gt_box9d: [K, 9] Matched ground-truth 3D boxes
        matched_pred_boxes9d: [K, 9] Matched predicted 3D boxes
    """
    matched_gt = []
    matched_pred = []
    used_preds = set()

    for i, gt_box in enumerate(gt_box2d):
        best_iou = 0
        best_j = -1
        for j, pred_box in enumerate(pred_box2d):
            if j in used_preds:
                continue
            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= match_iou and best_j != -1:
            matched_gt.append(gt_box9d[i])
            matched_pred.append(pred_boxes9d[best_j])
            used_preds.add(best_j)

    if matched_gt:
        return np.stack(matched_gt), np.stack(matched_pred)
    else:
        return np.zeros((0, 9)), np.zeros((0, 9))

def box9d_to_2d(boxes9d, img_width=1280., img_height=720., intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None):
    import numpy as np
    import cv2
    from scipy.spatial.transform import Rotation as R

    if intrinsic_mat is None:
        intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                  [0, img_height, img_height / 2],
                                  [0, 0, 1]], dtype=np.float32)

    if extrinsic_mat is None:
        extrinsic_mat = np.eye(4)

    if distortion_matrix is None:
        distortion_matrix = np.zeros(5, dtype=np.float32)

    corners_local = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ], dtype=np.float32)

    boxes2d = []
    size = []

    for box in boxes9d:
        x, y, z, l, w, h, angle1, angle2, angle3 = box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7],box[8]

        corners = corners_local * np.array([l, w, h])
        rotation_matrix = R.from_euler('xyz', [angle1, angle2, angle3], degrees=False).as_matrix()
        corners = np.dot(corners, rotation_matrix.T)
        corners += np.array([x, y, z])

        # ---- 变换到相机坐标系 ----
        corners_homogeneous = np.hstack((corners, np.ones((corners.shape[0], 1))))
        corners_camera = (extrinsic_mat @ corners_homogeneous.T).T[:, :3]

        # ---- 投影到图像 ----
        rvec, _ = cv2.Rodrigues(extrinsic_mat[:3, :3])
        tvec = extrinsic_mat[:3, 3]

        corners_2d, _ = cv2.projectPoints(corners, rvec, tvec, intrinsic_mat, distortion_matrix)
        corners_2d = corners_2d.reshape(-1, 2).astype(int)

        corners_2d[:, 0] = np.clip(corners_2d[:, 0], a_min=0, a_max=int(img_width)-1)
        corners_2d[:, 1] = np.clip(corners_2d[:, 1], a_min=0, a_max=int(img_height)-1)

        x1, y1 = corners_2d.min(axis=0)
        x2, y2 = corners_2d.max(axis=0)
        boxes2d.append([x1, y1, x2, y2])
        size.append(min(abs(x2 - x1), abs(y2 - y1)))

    return np.array(boxes2d), np.array(size)

def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5):
    """
    Perform Non-Maximum Suppression using NumPy.

    Args:
        boxes (np.ndarray): Nx4 array of boxes in [x1, y1, x2, y2] format.
        scores (np.ndarray): Confidence scores of the boxes.
        iou_threshold (float): IOU threshold for suppression.

    Returns:
        keep (List[int]): List of indices of boxes to keep.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # sort descending

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU of the top-scoring box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]  # shift by 1 due to indexing into order[1:]

    return keep


def draw_box9d_on_image_with_info(boxes9d, image, info, img_width=1920., img_height=1080., color=(255, 0, 0),
                                  intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None,
                                  offset=np.array([0, 0, 0])):
    """
    Draw multiple 9-DoF 3D boxes on the image, considering camera distortion parameters, with info text.

    :param boxes9d: (N, 9) Each box's parameters are (x, y, z, l, w, h, angle1, angle2, angle3)
    :param image: (H, W, 3) Input image
    :param info: (N,) Array of strings containing info to display for each box
    :param img_width: Image width
    :param img_height: Image height
    :param color: Box color (B, G, R)
    :param intrinsic_mat: (3, 3) Camera intrinsic matrix
    :param extrinsic_mat: (4, 4) Camera extrinsic matrix
    :param distortion_matrix: (5, ) Camera distortion parameters [k1, k2, p1, p2, k3]
    :param offset: (3,) Offset to apply to all boxes
    :return: Image with 3D boxes and info text drawn (H, W, 3)
    """
    if intrinsic_mat is None:
        intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                  [0, img_height, img_height / 2],
                                  [0, 0, 1]], dtype=np.float32)

    if extrinsic_mat is None:
        extrinsic_mat = np.eye(4)

    if distortion_matrix is None:
        distortion_matrix = np.zeros(5, dtype=np.float32)


    info = [str(x)[0:5] for x in info]

    # Define the 8 corner points of the 3D box (relative to the center point)
    corners_local = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ], dtype=np.float32)

    corners_local += offset

    for i, box in enumerate(boxes9d):
        x, y, z, l, w, h, angle1, angle2, angle3 = box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], box[
            8]

        # Scale the corner points to match the box dimensions
        corners = corners_local * np.array([l, w, h])

        # Rotate the corner points
        rotation_matrix = R.from_euler('xyz', [angle1, angle2, angle3], degrees=False).as_matrix()
        corners = np.dot(corners, rotation_matrix.T)

        # Translate the corner points to the box center
        corners += np.array([x, y, z])

        # Convert 3D points to homogeneous coordinates
        corners_homogeneous = np.hstack((corners, np.ones((corners.shape[0], 1))))

        # Project onto the image plane (considering distortion)
        corners_2d, _ = cv2.projectPoints(corners, extrinsic_mat[:3, :3], extrinsic_mat[:3, 3], intrinsic_mat,
                                          distortion_matrix)
        corners_2d = corners_2d.reshape(-1, 2).astype(int)

        # Draw the edges of the 3D box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)  # Side faces
        ]
        for edge in edges:
            start, end = edge
            cv2.line(image, tuple(corners_2d[start]), tuple(corners_2d[end]), color, 2)

        # Calculate top center point for text placement
        top_center_local = np.array([0, 0, 0.5]) * np.array([l, w, h])
        top_center_rotated = np.dot(top_center_local, rotation_matrix.T)
        top_center_world = top_center_rotated + np.array([x, y, z])

        # Project top center point to 2D
        top_center_2d, _ = cv2.projectPoints(top_center_world.reshape(1, 3), extrinsic_mat[:3, :3],
                                             extrinsic_mat[:3, 3], intrinsic_mat, distortion_matrix)
        top_center_2d = top_center_2d.reshape(-1, 2).astype(int)[0]

        # Draw info text
        text = str(info[i]) if i < len(info) else ""
        font_scale = 0.5
        thickness = 1
        # Get text size to adjust position
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Adjust text position to be centered above the top center point
        text_x = top_center_2d[0] - text_width // 2
        text_y = top_center_2d[1] - 10  # 10 pixels above the point
        # Ensure text is within image bounds
        text_x = max(0, min(text_x, image.shape[1] - text_width))
        text_y = max(text_height, min(text_y, image.shape[0]))

        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return image


class LAA3D_Det_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training, root_path, logger):
        super(LAA3D_Det_Dataset, self).__init__(dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger)

        self.dataset_cfg=dataset_cfg
        self.root_path=root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.training=training
        self.logger=logger

        self.raw_im_width = dataset_cfg.IM_SIZE[0]
        self.raw_im_hight = dataset_cfg.IM_SIZE[1]

        self.new_im_width = dataset_cfg.IM_RESIZE[0]
        self.new_im_hight = dataset_cfg.IM_RESIZE[1]

        self.distortion_matrix = np.array([0, 0, 0, 0, 0])

        self.im_num = self.dataset_cfg.IM_NUM

        self.center_rad = self.dataset_cfg.CENTER_RAD

        self.stride = self.dataset_cfg.STRIDE

        self.class_name_dict = self.dataset_cfg.CLASS_NAMES

        self.interval= self.dataset_cfg.SAMPLED_INTERVAL[self.mode]


        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.data_info = None

        with open(os.path.join(self.root_path, self.split+'_info.pkl'), 'rb') as f:
            self.data_info = pkl.load(f)

        data_info = []

        for i in range(0,len(self.data_info), self.interval):
            data_info.append(self.data_info[i])

        self.data_info = data_info

    def set_split(self, split):
        self.split = split

        self.data_info = None

        with open(os.path.join(self.root_path, self.split + '_info.pkl'), 'rb') as f:
            self.data_info = pkl.load(f)

        data_info = []

        for i in range(0, len(self.data_info), self.interval):
            data_info.append(self.data_info[i])

        self.data_info = data_info


    def __len__(self):

        return len(self.data_info)

    def __getitem__(self, item):

        # item = item%10

        info = self.data_info[item]

        setting_id = info['setting_id']
        seq_id = info['seq_id']
        brightness = info['brightness']
        frame_id = info['frame_id']
        relative_im_path = info['relative_im_path'].replace('\\','/')



        annos = info['annos']

        boxes9d = annos['box']  # 3D boxes
        this_boxes_2d = annos['box2d']  # 2D boxes

        diff = annos['diff']  # difficulty level
        this_coarse_class = annos['coarse_class']  # object classes
        this_fine_class = annos['fine_class']  # object classes
        this_id = annos['ob_id']  # object identity

        boxes9d = boxes9d[diff < self.dataset_cfg.DiffLevel]
        this_coarse_class = this_coarse_class[diff < self.dataset_cfg.DiffLevel]
        this_boxes_2d = this_boxes_2d[diff < self.dataset_cfg.DiffLevel]
        diff = diff[diff < self.dataset_cfg.DiffLevel]

        image_info = info['image_info']
        intrinsic = image_info['intrinsic']
        extrinsic = image_info['extrinsic']

        image = cv2.imread(os.path.join(self.root_path, relative_im_path)).astype(np.float32)

        random_frame_id = np.random.randint(0,len(self.data_info)-1)
        random_info = self.data_info[random_frame_id]
        random_relative_im_path = random_info['relative_im_path'].replace('\\', '/')
        random_image = cv2.imread(os.path.join(self.root_path, random_relative_im_path)).astype(np.float32)
        random_boxes_2d = random_info['annos']['box2d']

        if len(boxes9d) == 0:
            boxes9d = np.empty(shape=(0, 9))
            this_coarse_class = np.empty(shape=(0,))
            diff = np.empty(shape=(0,))
            this_boxes_2d = np.empty(shape=(0, 4))

        if len(random_boxes_2d) == 0:
            random_boxes_2d = np.empty(shape=(0, 4))

        aug_dict = {'image': image,
                    'random_image': random_image,
                    'intrinsic': intrinsic,
                    'extrinsic': extrinsic,
                    'boxes9d': boxes9d,
                    'this_boxes_2d': this_boxes_2d,
                    'random_boxes_2d': random_boxes_2d}

        if self.data_augmentor is not None and self.training:
            aug_dict = self.data_augmentor(aug_dict)

            image = aug_dict['image']
            intrinsic = aug_dict['intrinsic']
            boxes9d = aug_dict['boxes9d']
            this_boxes_2d = aug_dict['this_boxes_2d']

            # image = draw_box9d_on_image(boxes9d, image,
            #                             img_width=self.raw_im_width,
            #                             img_height=self.raw_im_hight,
            #                             color=(0, 0, 255),
            #                             intrinsic_mat=intrinsic,
            #                             extrinsic_mat=extrinsic,
            #                             distortion_matrix=self.distortion_matrix,
            #                             offset=np.array([0., 0., 0.]))
            # cv2.imwrite('image/'+frame_id+'.png', image)

        resized_img = cv2.resize(
            image,
            (self.new_im_width, self.new_im_hight),  # 目标尺寸 (width, height)
            interpolation=cv2.INTER_AREA  # 缩小推荐使用区域插值
        )
        resized_img = resized_img.transpose(2,0,1)
        C,W,H = resized_img.shape
        resized_img = resized_img.reshape(1,C,W,H)

        boxes9d = np.array(boxes9d)

        data_dict = {}

        data_dict['intrinsic'] = np.array([intrinsic])
        data_dict['extrinsic'] = np.array([extrinsic])
        data_dict['distortion'] = np.array([self.distortion_matrix])
        data_dict['raw_im_size'] = np.array([self.raw_im_width, self.raw_im_hight])
        data_dict['new_im_size'] = np.array([self.new_im_width, self.new_im_hight])
        data_dict['seq_id'] = seq_id
        data_dict['frame_id'] = frame_id
        data_dict['setting_id'] = setting_id
        data_dict['brightness'] = brightness
        data_dict['relative_im_path'] = relative_im_path
        data_dict['stride'] = self.stride
        data_dict['image'] = resized_img
        data_dict['gt_box9d'] = boxes9d
        data_dict['this_boxes_2d'] = this_boxes_2d
        data_dict['gt_diff'] = diff
        data_dict['gt_name'] = this_coarse_class

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
            setting_id = batch_dict['setting_id'][batch_id]
            brightness = batch_dict['brightness'][batch_id]
            relative_im_path = batch_dict['relative_im_path'][batch_id]

            intrinsic = batch_dict['intrinsic'][batch_id][0] # 3, 3
            extrinsic = batch_dict['extrinsic'][batch_id][0] # 4, 4
            distortion = batch_dict['distortion'][batch_id][0] # 5,
            raw_im_size = batch_dict['raw_im_size'][batch_id] # 2,
            new_im_size = batch_dict['new_im_size'][batch_id] # 2,

            image = batch_dict['image'][batch_id][0].cpu().numpy()

            image = image.transpose(1,2,0)*255

            # key_points_2d = batch_dict['key_points_2d'][batch_id]
            confidence = batch_dict['confidence'][batch_id]
            im_path = os.path.join(self.root_path, relative_im_path)

            pred_name = self.name_from_code(pred_boxes9d[:,-1])

            frame_dict = {'seq_id': seq_id,
                          'frame_id': frame_id,
                          'setting_id': setting_id,
                          'brightness': brightness,
                          'intrinsic': intrinsic,
                          'extrinsic': extrinsic,
                          'distortion': distortion,
                          'raw_im_size': raw_im_size,
                          'confidence': confidence,
                          'pred_box9d': pred_boxes9d,
                          'pred_name': pred_name,
                          'im_path': im_path
                          }

            if 'gt_box9d' in batch_dict:
                gt_diff = batch_dict['gt_diff'][batch_id]
                frame_dict['gt_diff'] = gt_diff
                gt_box9d = batch_dict['gt_box9d'][batch_id].cpu().numpy() # 1, 9
                frame_dict['gt_box9d'] = gt_box9d
                gt_name = batch_dict['gt_name'][batch_id]
                frame_dict['gt_name'] = gt_name

            annos.append(frame_dict)


            #####################################

            image = cv2.imread(im_path)

            # image = cv2.flip(image, 1)

            # pred_boxes9d = pred_boxes9d[pred_name=='MAV']

            z = gt_box9d.__len__() - 1

            while z > 0 and gt_box9d[z].sum() == 0:
                z -= 1

            gt_box9d = gt_box9d[:z + 1]
            # gt_box9d = gt_box9d[gt_name=='MAV']


            pred_box2d, _ = box9d_to_2d(pred_boxes9d, intrinsic_mat=intrinsic, extrinsic_mat=extrinsic, distortion_matrix=distortion)
            gt_box2d, _ = box9d_to_2d(gt_box9d, intrinsic_mat=intrinsic, extrinsic_mat=extrinsic, distortion_matrix=distortion)

            gt_box9d_new, pred_boxes9d_new = match_gt(gt_box9d, pred_boxes9d, gt_box2d, pred_box2d)

            dis_error = np.linalg.norm(gt_box9d_new[:,0:3] - pred_boxes9d_new[:,0:3], axis=-1)

            image = draw_box9d_on_image_with_info(pred_boxes9d_new, image, dis_error,
                                        img_width=raw_im_size[0],
                                        img_height=raw_im_size[1],
                                        color=(255, 0, 0),
                                        intrinsic_mat=intrinsic,
                                        extrinsic_mat=extrinsic,
                                        distortion_matrix=distortion,
                                        offset=np.array([0., 0., 0.]))

            image = draw_box9d_on_image(gt_box9d_new, image,
                                        img_width=raw_im_size[0],
                                        img_height=raw_im_size[1],
                                        color=(0, 0, 255),
                                        intrinsic_mat=intrinsic,
                                        extrinsic_mat=extrinsic,
                                        distortion_matrix=distortion,
                                        offset=np.array([0., 0., 0.]))

            os.makedirs('image',exist_ok=True)

            cv2.imwrite('image/'+seq_id+frame_id+'.png', image)

            ####################################

        return annos




    def evaluation_all(self, annos, metric_root_path):

        laa_ads_fun = LAA3D_ADS_Metric(eval_config = self.dataset_cfg.LAA3D_ADS_METRIC, classes = self.class_name_dict, metric_save_path=metric_root_path)

        final_str = laa_ads_fun.eval(annos)

        return final_str


    def evaluation_private(self, annos, metric_root_path):

        MetricPrivate = self.dataset_cfg.MetricPrivate

        setting_mask = MetricPrivate.SetttingMask
        wheather_mask = MetricPrivate.WheatherMask
        brightness_mask = MetricPrivate.BrightnessMask
        drop_rain = MetricPrivate.DropRain

        new_annos = []

        for info in annos:
            setting_id = info['setting_id']
            seq_id = info['seq_id']
            brightness = info['brightness']
            frame_id = info['frame_id']

            if len(setting_mask)>0:
                setting_flag = False
                for each_name in setting_mask:
                    if each_name in setting_id:
                        setting_flag=True
            else:
                setting_flag = True

            if len(wheather_mask)>0:
                wheather_flag = False
                for each_name in wheather_mask:
                    if each_name in seq_id:
                        wheather_flag=True
            else:
                wheather_flag = True

            if drop_rain and 'rain' in seq_id:
                drop_rain_flag = False
            else:
                drop_rain_flag = True

            if brightness > brightness_mask[0] and brightness < brightness_mask[1]:
                brightness_flag = True
            else:
                brightness_flag = False

            if setting_flag and wheather_flag and drop_rain_flag and brightness_flag:
                new_annos.append(info)

        laa_ads_fun = LAA3D_ADS_Metric(eval_config = self.dataset_cfg.LAA3D_ADS_METRIC, classes = self.class_name_dict, metric_save_path=metric_root_path)

        final_str = laa_ads_fun.eval(new_annos)

        return final_str

    def evaluation(self, annos, metric_root_path):

        print('evaluating !!!')

        i_str = self.evaluation_private(annos,metric_root_path)

        return i_str