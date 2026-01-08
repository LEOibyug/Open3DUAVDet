
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import torch
from uavdet3d.utils.centernet_utils import draw_gaussian_to_heatmap, draw_res_to_heatmap
import copy
import time


def projectPoints(points_cam, intrinsic_mat):
    """
    相机坐标系 → 图像坐标
    points_cam: (N,3) 相机坐标系下的三维点
    intrinsic_mat: (3,3) 相机内参矩阵
    """
    # 提取内参
    fx = intrinsic_mat[0, 0]
    fy = intrinsic_mat[1, 1]
    cx = intrinsic_mat[0, 2]
    cy = intrinsic_mat[1, 2]

    # 使用cv2.projectPoints (需要外参为零)
    rvec = np.zeros(3)  # 零旋转
    tvec = np.zeros(3)  # 零平移
    camera_matrix = intrinsic_mat
    dist_coeffs = None  # 不考虑畸变

    points_2d, _ = cv2.projectPoints(points_cam, rvec, tvec, camera_matrix, dist_coeffs)
    points_2d = points_2d.reshape(-1, 2)

    depth = points_cam[:, 2]
    return points_2d, depth


def backProject(points_2d, depth, intrinsic_mat):
    """
    图像坐标 + 深度 → 相机坐标系
    points_2d: (N,2) 像素坐标
    depth: (N,) 深度值
    intrinsic_mat: (3,3) 相机内参矩阵
    """
    # 提取内参
    fx = intrinsic_mat[0, 0]
    fy = intrinsic_mat[1, 1]
    cx = intrinsic_mat[0, 2]
    cy = intrinsic_mat[1, 2]

    # 使用cv2.undistortPoints获取归一化坐标
    points_2d = points_2d.reshape(-1, 1, 2).astype(np.float32)
    points_norm = cv2.undistortPoints(points_2d, intrinsic_mat, None)
    points_norm = points_norm.reshape(-1, 2)

    # 反投影到相机坐标系
    x_cam = points_norm[:, 0] * depth
    y_cam = points_norm[:, 1] * depth
    z_cam = depth

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    return points_cam


def key_point_encoder(boxes9d, encode_corner=None, intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None, offset=np.array([-0.15, 0.05, 0])):

    corners_local = np.array(encode_corner, dtype=np.float32)

    corners_local += offset

    all_corners_3d = []
    all_corners_2d = []

    for box in boxes9d:
        x, y, z, l, w, h, angle1, angle2, angle3 = box

        # Scale the corner points to match the box dimensions
        corners = corners_local * np.array([l, w, h])

        # Rotate the corner points
        rotation_matrix = R.from_euler('xyz', [angle1, angle2, angle3], degrees=False).as_matrix()
        corners = np.dot(corners, rotation_matrix.T)

        # Translate the corner points to the box center
        corners += np.array([x, y, z])

        corners_2d, _ = cv2.projectPoints(corners, extrinsic_mat[:3, :3], extrinsic_mat[:3, 3], intrinsic_mat,
                                          distortion_matrix)
        corners_2d = corners_2d.reshape(-1, 2)

        all_corners_3d.append(corners)
        all_corners_2d.append(corners_2d)

    return np.array(all_corners_3d), np.array(all_corners_2d)

def key_point_decoder( encode_corner,
                       off_set,
                       pred_heat_map=None, # 1,1,4,W,H
                       pred_res_x=None, # 1,1,4,W,H
                       pred_res_y=None, # 1,1,4,W,H
                       new_im_width=None,
                       new_im_hight=None,
                       raw_im_width=None,
                       raw_im_hight=None,
                       stride=None,
                       im_num=None,
                       obj_num=None,
                       size=None,
                       intrinsic=None,
                       distortion=None,
                       PnP_algo = 'SOLVEPNP_EPNP'):
    im_num = im_num
    obj_num = obj_num

    corners_local = np.array(encode_corner)+off_set

    key_pts_num = len(corners_local)

    key_points_2d = torch.zeros(im_num, obj_num, key_pts_num, 2)

    confidence = torch.zeros(im_num, obj_num)

    for im_id in range(im_num):
        for ob_id in range(obj_num):

            all_conf = []
            for k_id in range(key_pts_num):
                this_heatmap = pred_heat_map[im_id, ob_id, k_id]
                this_res_x = pred_res_x[im_id, ob_id, k_id]
                this_res_y = pred_res_y[im_id, ob_id, k_id]

                shape_map = this_heatmap.shape

                flat_x = this_heatmap.flatten()
                values, linear_indices = torch.topk(flat_x, k=1)

                c_num = shape_map[-1]
                rows = linear_indices // c_num
                cols = linear_indices % c_num

                row = rows[0]
                col = cols[0]
                conf = values[0]
                all_conf.append(conf)

                res_x = this_res_x[row.long(), col.long()]
                res_y = this_res_y[row.long(), col.long()]

                y_cor = row.float() + res_y
                x_cor = col.float() + res_x

                delta0 = (new_im_width / raw_im_width / stride)
                delta1 = (new_im_hight / raw_im_hight / stride)

                key_points_2d[im_id, ob_id, k_id, 0] = (x_cor / delta0)
                key_points_2d[im_id, ob_id, k_id, 1] = (y_cor / delta1)

            confidence[im_id, ob_id] = torch.mean(torch.stack(all_conf))

    key_points_2d = key_points_2d.detach().cpu().numpy()
    confidence = confidence.detach().cpu().numpy()

    all_pred_box9d = []

    for im_id in range(im_num):

        all_obj_each_im = []

        for ob_id in range(obj_num):
            # Scale the local corner points according to the target size
            this_corner = corners_local * size[ob_id]

            im_pts = key_points_2d[im_id, ob_id]

            if PnP_algo == 'SOLVEPNP_EPNP':
                success, rvec, tvec = cv2.solvePnP(this_corner, np.array(im_pts, dtype=np.float64), intrinsic[im_id],
                                                   distortion[im_id], flags=cv2.SOLVEPNP_EPNP)  # , flags=cv2.SOLVEPNP_AP3P
            elif PnP_algo == 'SOLVEPNP_AP3P':
                success, rvec, tvec = cv2.solvePnP(this_corner, np.array(im_pts, dtype=np.float64), intrinsic[im_id],
                                                   distortion[im_id],
                                                   flags=cv2.SOLVEPNP_AP3P)  # , flags=cv2.SOLVEPNP_AP3P
            else:
                success, rvec, tvec = cv2.solvePnP(this_corner, np.array(im_pts, dtype=np.float64), intrinsic[im_id],
                                                   distortion[im_id],
                                                   flags=cv2.SOLVEPNP_ITERATIVE)  # , flags=cv2.SOLVEPNP_AP3P

            if success:
                # Convert the rotation vector to a rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                # Extract Euler angles (rotation angles around x, y, z axes) from the rotation matrix
                sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                angle1 = np.arctan2(R[2, 1], R[2, 2])  # Rotation around x-axis
                angle2 = np.arctan2(-R[2, 0], sy)  # Rotation around y-axis
                angle3 = np.arctan2(R[1, 0], R[0, 0])  # Rotation around z-axis

                # Combine the translation vector and rotation angles into 9D parameters
                box9d = [tvec[0, 0], tvec[1, 0], tvec[2, 0], size[ob_id][0], size[ob_id][1], size[ob_id][2], angle1, angle2, angle3]
                all_obj_each_im.append(box9d)
            else:
                # If PnP solving fails, return default values
                all_obj_each_im.append([0, 0, 0, size[ob_id][0], size[ob_id][1], size[ob_id][2], 0, 0, 0])

        all_pred_box9d.append(all_obj_each_im)

    all_pred_box9d = np.array(all_pred_box9d)

    pred_boxes9d = all_pred_box9d.reshape(obj_num*im_num, 9)

    key_points_2d = key_points_2d.reshape(obj_num*im_num, key_pts_num, 2),
    confidence = confidence.reshape(obj_num*im_num)


    return key_points_2d, confidence, pred_boxes9d


def center_point_encoder(gt_box9d_with_cls,
                           intrinsic_mat,
                           extrinsic_mat,
                           distortion_matrix,
                           new_im_width=None,
                           new_im_hight=None,
                           raw_im_width=None,
                           raw_im_hight=None,
                           stride=None,
                           im_num=None,
                           class_name_config=None,
                           center_rad=None):


    cls_num = len(class_name_config)

    gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = [], [], [], [], []

    for im_id in range(im_num):
        in_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]


        xyz = copy.deepcopy(gt_box9d_with_cls[:,0:3])

        ptx_im, depth = projectPoints(xyz,  in_mat)#
        ptx_im = ptx_im.reshape(-1,2)
        depth = depth.reshape(-1,1)

        this_heat_map = torch.zeros(cls_num, new_im_hight // stride, new_im_width // stride)

        this_res_map = np.zeros(shape = (2, new_im_hight // stride, new_im_width // stride))

        this_dis_map = np.zeros(shape = (1, new_im_hight // stride, new_im_width // stride))

        this_size_map = np.zeros(shape = (3, new_im_hight // stride, new_im_width // stride))

        this_angle_map = np.zeros(shape = (6, new_im_hight // stride, new_im_width // stride))


        for obj_i, obj in enumerate(gt_box9d_with_cls):

            this_cls = int(obj[-1])

            center = ptx_im[obj_i]

            dis = depth[obj_i]

            center[0] *= (new_im_width / raw_im_width / stride)
            center[1] *= (new_im_hight / raw_im_hight / stride)

            this_heat_map[this_cls] = draw_gaussian_to_heatmap(this_heat_map[this_cls], center, center_rad)

            this_res_map[0], this_res_map[1] = draw_res_to_heatmap(this_res_map[0], this_res_map[1], center)

            try:

                this_dis_map[0, int(center[1]), int(center[0])] = dis

                l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]

                this_size_map[0, int(center[1]), int(center[0])] = l
                this_size_map[1, int(center[1]), int(center[0])] = w
                this_size_map[2, int(center[1]), int(center[0])] = h

                this_angle_map[0, int(center[1]), int(center[0])] = np.cos(a1)
                this_angle_map[1, int(center[1]), int(center[0])] = np.sin(a1)
                this_angle_map[2, int(center[1]), int(center[0])] = np.cos(a2)
                this_angle_map[3, int(center[1]), int(center[0])] = np.sin(a2)
                this_angle_map[4, int(center[1]), int(center[0])] = np.cos(a3)
                this_angle_map[5, int(center[1]), int(center[0])] = np.sin(a3)

            except:

                continue

        gt_hm.append(this_heat_map.cpu().numpy())
        gt_center_res.append(this_res_map)
        gt_center_dis.append(this_dis_map)
        gt_dim.append(this_size_map)
        gt_rot.append(this_angle_map)

    return np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dis), np.array(gt_dim), np.array(gt_rot)


def center_point_encoder_DORN(gt_box9d_with_cls,
                               intrinsic_mat,
                               extrinsic_mat,
                               distortion_matrix,
                               new_im_width=None,
                               new_im_hight=None,
                               raw_im_width=None,
                               raw_im_hight=None,
                               stride=None,
                               im_num=None,
                               class_name_config=None,
                               center_rad=None,
                               depth_num = None,
                               depth_interval = None):


        cls_num = len(class_name_config)

        gt_hm, gt_center_res, gt_center_dis_bin, gt_center_dis_res, gt_dim, gt_rot = [], [], [], [], [], []

        for im_id in range(im_num):
            in_mat = intrinsic_mat[im_id]
            ex_mat = extrinsic_mat[im_id]
            dis_mat = distortion_matrix[im_id]


            xyz = copy.deepcopy(gt_box9d_with_cls[:,0:3])

            ptx_im, depth = projectPoints(xyz, in_mat)#
            ptx_im = ptx_im.reshape(-1,2)
            depth = depth.reshape(-1,1)

            this_heat_map = torch.zeros(cls_num, new_im_hight // stride, new_im_width // stride)

            this_res_map = np.zeros(shape = (2, new_im_hight // stride, new_im_width // stride))

            this_dis_bin_map = np.zeros(shape = (depth_num, new_im_hight // stride, new_im_width // stride))
            this_dis_res_map = np.zeros(shape = (depth_num, new_im_hight // stride, new_im_width // stride))

            this_size_map = np.zeros(shape = (3, new_im_hight // stride, new_im_width // stride))

            this_angle_map = np.zeros(shape = (6, new_im_hight // stride, new_im_width // stride))


            for obj_i, obj in enumerate(gt_box9d_with_cls):

                this_cls = int(obj[-1])

                center = ptx_im[obj_i]

                dis = depth[obj_i]

                center[0] *= (new_im_width / raw_im_width / stride)
                center[1] *= (new_im_hight / raw_im_hight / stride)

                this_heat_map[this_cls] = draw_gaussian_to_heatmap(this_heat_map[this_cls], center, center_rad)

                this_res_map[0], this_res_map[1] = draw_res_to_heatmap(this_res_map[0], this_res_map[1], center)

                try:

                    dis_grid_id = int(dis/depth_interval)
                    dis_residual = dis%depth_interval

                    if dis_grid_id>=depth_num:
                        dis_grid_id = depth_num-1
                        dis_residual = dis-(depth_num-1)*depth_interval


                    this_dis_bin_map[dis_grid_id, int(center[1]), int(center[0])] = 1

                    this_dis_res_map[dis_grid_id, int(center[1]), int(center[0])] = dis_residual

                    l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]

                    this_size_map[0, int(center[1]), int(center[0])] = l
                    this_size_map[1, int(center[1]), int(center[0])] = w
                    this_size_map[2, int(center[1]), int(center[0])] = h

                    this_angle_map[0, int(center[1]), int(center[0])] = np.cos(a1)
                    this_angle_map[1, int(center[1]), int(center[0])] = np.sin(a1)
                    this_angle_map[2, int(center[1]), int(center[0])] = np.cos(a2)
                    this_angle_map[3, int(center[1]), int(center[0])] = np.sin(a2)
                    this_angle_map[4, int(center[1]), int(center[0])] = np.cos(a3)
                    this_angle_map[5, int(center[1]), int(center[0])] = np.sin(a3)

                except:

                    continue

            gt_hm.append(this_heat_map.cpu().numpy())
            gt_center_res.append(this_res_map)
            gt_center_dis_bin.append(this_dis_bin_map)
            gt_center_dis_res.append(this_dis_res_map)
            gt_dim.append(this_size_map)
            gt_rot.append(this_angle_map)

        return np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dis_bin), np.array(gt_center_dis_res), np.array(gt_dim), np.array(gt_rot)


def center_point_encoder_DORN_focus_640(gt_box9d_with_cls,
                              intrinsic_mat,
                              extrinsic_mat,
                              distortion_matrix,
                              new_im_width=None,
                              new_im_hight=None,
                              raw_im_width=None,
                              raw_im_hight=None,
                              stride=None,
                              im_num=None,
                              class_name_config=None,
                              center_rad=None,
                              depth_num=None,
                              depth_interval=None):
    cls_num = len(class_name_config)

    gt_hm, gt_center_res, gt_center_dis_bin, gt_center_dis_res, gt_dim, gt_rot = [], [], [], [], [], []



    for im_id in range(im_num):
        in_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]

        defult_focus = 640

        focus_scale = in_mat[0, 0] / defult_focus

        gt_box9d_with_cls2 = copy.deepcopy(gt_box9d_with_cls)

        gt_box9d_with_cls2[:, 2] = gt_box9d_with_cls[:, 2] / focus_scale

        new_in_mat = copy.deepcopy(in_mat)

        new_in_mat[0:2,0:2] = in_mat[0:2,0:2] /focus_scale

        xyz = copy.deepcopy(gt_box9d_with_cls2[:, 0:3])

        ptx_im, depth = projectPoints(xyz,new_in_mat)  #
        ptx_im = ptx_im.reshape(-1, 2)
        depth = depth.reshape(-1, 1)

        this_heat_map = torch.zeros(cls_num, new_im_hight // stride, new_im_width // stride)

        this_res_map = np.zeros(shape=(2, new_im_hight // stride, new_im_width // stride))

        this_dis_bin_map = np.zeros(shape=(depth_num, new_im_hight // stride, new_im_width // stride))
        this_dis_res_map = np.zeros(shape=(depth_num, new_im_hight // stride, new_im_width // stride))

        this_size_map = np.zeros(shape=(3, new_im_hight // stride, new_im_width // stride))

        this_angle_map = np.zeros(shape=(6, new_im_hight // stride, new_im_width // stride))

        for obj_i, obj in enumerate(gt_box9d_with_cls2):

            this_cls = int(obj[-1])

            center = ptx_im[obj_i]


            dis = depth[obj_i]

            center[0] *= (new_im_width / raw_im_width / stride)
            center[1] *= (new_im_hight / raw_im_hight / stride)

            this_heat_map[this_cls] = draw_gaussian_to_heatmap(this_heat_map[this_cls], center, center_rad)

            this_res_map[0], this_res_map[1] = draw_res_to_heatmap(this_res_map[0], this_res_map[1], center)

            try:

                dis_grid_id = int(dis / depth_interval)
                dis_residual = dis % depth_interval

                if dis_grid_id >= depth_num:
                    dis_grid_id = depth_num - 1
                    dis_residual = dis - (depth_num - 1) * depth_interval

                this_dis_bin_map[dis_grid_id, int(center[1]), int(center[0])] = 1

                this_dis_res_map[dis_grid_id, int(center[1]), int(center[0])] = dis_residual

                l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]

                this_size_map[0, int(center[1]), int(center[0])] = l
                this_size_map[1, int(center[1]), int(center[0])] = w
                this_size_map[2, int(center[1]), int(center[0])] = h

                this_angle_map[0, int(center[1]), int(center[0])] = np.cos(a1)
                this_angle_map[1, int(center[1]), int(center[0])] = np.sin(a1)
                this_angle_map[2, int(center[1]), int(center[0])] = np.cos(a2)
                this_angle_map[3, int(center[1]), int(center[0])] = np.sin(a2)
                this_angle_map[4, int(center[1]), int(center[0])] = np.cos(a3)
                this_angle_map[5, int(center[1]), int(center[0])] = np.sin(a3)

            except:

                continue

        gt_hm.append(this_heat_map.cpu().numpy())
        gt_center_res.append(this_res_map)
        gt_center_dis_bin.append(this_dis_bin_map)
        gt_center_dis_res.append(this_dis_res_map)
        gt_dim.append(this_size_map)
        gt_rot.append(this_angle_map)

    return np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dis_bin), np.array(gt_center_dis_res), np.array(
        gt_dim), np.array(gt_rot)


def center_point_encoder_DORN_focus_1280(gt_box9d_with_cls,
                              intrinsic_mat,
                              extrinsic_mat,
                              distortion_matrix,
                              new_im_width=None,
                              new_im_hight=None,
                              raw_im_width=None,
                              raw_im_hight=None,
                              stride=None,
                              im_num=None,
                              class_name_config=None,
                              center_rad=None,
                              depth_num=None,
                              depth_interval=None):
    cls_num = len(class_name_config)

    gt_hm, gt_center_res, gt_center_dis_bin, gt_center_dis_res, gt_dim, gt_rot = [], [], [], [], [], []



    for im_id in range(im_num):
        in_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]

        defult_focus = 1280

        focus_scale = in_mat[0, 0] / defult_focus

        gt_box9d_with_cls2 = copy.deepcopy(gt_box9d_with_cls)

        gt_box9d_with_cls2[:, 2] = gt_box9d_with_cls[:, 2] / focus_scale

        new_in_mat = copy.deepcopy(in_mat)

        new_in_mat[0:2,0:2] = in_mat[0:2,0:2] /focus_scale

        xyz = copy.deepcopy(gt_box9d_with_cls2[:, 0:3])

        ptx_im, depth = projectPoints(xyz, new_in_mat)  #
        ptx_im = ptx_im.reshape(-1, 2)
        depth = depth.reshape(-1, 1)

        this_heat_map = torch.zeros(cls_num, new_im_hight // stride, new_im_width // stride)

        this_res_map = np.zeros(shape=(2, new_im_hight // stride, new_im_width // stride))

        this_dis_bin_map = np.zeros(shape=(depth_num, new_im_hight // stride, new_im_width // stride))
        this_dis_res_map = np.zeros(shape=(depth_num, new_im_hight // stride, new_im_width // stride))

        this_size_map = np.zeros(shape=(3, new_im_hight // stride, new_im_width // stride))

        this_angle_map = np.zeros(shape=(6, new_im_hight // stride, new_im_width // stride))

        for obj_i, obj in enumerate(gt_box9d_with_cls2):

            this_cls = int(obj[-1])

            center = ptx_im[obj_i]


            dis = depth[obj_i]

            center[0] *= (new_im_width / raw_im_width / stride)
            center[1] *= (new_im_hight / raw_im_hight / stride)

            this_heat_map[this_cls] = draw_gaussian_to_heatmap(this_heat_map[this_cls], center, center_rad)

            this_res_map[0], this_res_map[1] = draw_res_to_heatmap(this_res_map[0], this_res_map[1], center)

            try:

                dis_grid_id = int(dis / depth_interval)
                dis_residual = dis % depth_interval

                if dis_grid_id >= depth_num:
                    dis_grid_id = depth_num - 1
                    dis_residual = dis - (depth_num - 1) * depth_interval

                this_dis_bin_map[dis_grid_id, int(center[1]), int(center[0])] = 1

                this_dis_res_map[dis_grid_id, int(center[1]), int(center[0])] = dis_residual

                l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]

                this_size_map[0, int(center[1]), int(center[0])] = l
                this_size_map[1, int(center[1]), int(center[0])] = w
                this_size_map[2, int(center[1]), int(center[0])] = h

                this_angle_map[0, int(center[1]), int(center[0])] = np.cos(a1)
                this_angle_map[1, int(center[1]), int(center[0])] = np.sin(a1)
                this_angle_map[2, int(center[1]), int(center[0])] = np.cos(a2)
                this_angle_map[3, int(center[1]), int(center[0])] = np.sin(a2)
                this_angle_map[4, int(center[1]), int(center[0])] = np.cos(a3)
                this_angle_map[5, int(center[1]), int(center[0])] = np.sin(a3)

            except:

                continue

        gt_hm.append(this_heat_map.cpu().numpy())
        gt_center_res.append(this_res_map)
        gt_center_dis_bin.append(this_dis_bin_map)
        gt_center_dis_res.append(this_dis_res_map)
        gt_dim.append(this_size_map)
        gt_rot.append(this_angle_map)

    return np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dis_bin), np.array(gt_center_dis_res), np.array(
        gt_dim), np.array(gt_rot)


def center_point_encoder_DORN_w2d(gt_box9d_with_cls,
                               this_boxes_2d,
                               intrinsic_mat,
                               extrinsic_mat,
                               distortion_matrix,
                               new_im_width=None,
                               new_im_hight=None,
                               raw_im_width=None,
                               raw_im_hight=None,
                               stride=None,
                               im_num=None,
                               class_name_config=None,
                               center_rad=None,
                               depth_num = None,
                               depth_interval = None):


    cls_num = len(class_name_config)

    gt_hm, gt_center_res, gt_center_dis_bin, gt_center_dis_res, gt_dim, gt_dim2d, gt_rot = [], [], [], [], [], [], []

    for im_id in range(im_num):
        in_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]


        xyz = copy.deepcopy(gt_box9d_with_cls[:,0:3])

        ptx_im, depth = projectPoints(xyz, in_mat)#
        ptx_im = ptx_im.reshape(-1,2)
        depth = depth.reshape(-1,1)

        this_heat_map = torch.zeros(cls_num, new_im_hight // stride, new_im_width // stride)

        this_res_map = np.zeros(shape = (2, new_im_hight // stride, new_im_width // stride))

        this_dis_bin_map = np.zeros(shape = (depth_num, new_im_hight // stride, new_im_width // stride))
        this_dis_res_map = np.zeros(shape = (depth_num, new_im_hight // stride, new_im_width // stride))

        this_size_map = np.zeros(shape = (3, new_im_hight // stride, new_im_width // stride))
        this_size_2d_map = np.zeros(shape=(2, new_im_hight // stride, new_im_width // stride))

        this_angle_map = np.zeros(shape = (6, new_im_hight // stride, new_im_width // stride))


        for obj_i, obj in enumerate(gt_box9d_with_cls):

            this_cls = int(obj[-1])

            center = ptx_im[obj_i]

            dis = depth[obj_i]

            box_2d = this_boxes_2d[obj_i]

            x_min, y_min, x_max, y_max = box_2d

            h_2d = x_max-x_min
            w_2d = y_max-y_min

            center[0] *= (new_im_width / raw_im_width / stride)
            center[1] *= (new_im_hight / raw_im_hight / stride)

            this_heat_map[this_cls] = draw_gaussian_to_heatmap(this_heat_map[this_cls], center, center_rad)

            this_res_map[0], this_res_map[1] = draw_res_to_heatmap(this_res_map[0], this_res_map[1], center)

            try:

                dis_grid_id = int(dis/depth_interval)
                dis_residual = dis%depth_interval

                if dis_grid_id>=depth_num:
                    dis_grid_id = depth_num-1
                    dis_residual = dis-(depth_num-1)*depth_interval


                this_dis_bin_map[dis_grid_id, int(center[1]), int(center[0])] = 1

                this_dis_res_map[dis_grid_id, int(center[1]), int(center[0])] = dis_residual

                l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]

                this_size_map[0, int(center[1]), int(center[0])] = l
                this_size_map[1, int(center[1]), int(center[0])] = w
                this_size_map[2, int(center[1]), int(center[0])] = h

                this_size_2d_map[0, int(center[1]), int(center[0])] = h_2d
                this_size_2d_map[1, int(center[1]), int(center[0])] = w_2d

                this_angle_map[0, int(center[1]), int(center[0])] = np.cos(a1)
                this_angle_map[1, int(center[1]), int(center[0])] = np.sin(a1)
                this_angle_map[2, int(center[1]), int(center[0])] = np.cos(a2)
                this_angle_map[3, int(center[1]), int(center[0])] = np.sin(a2)
                this_angle_map[4, int(center[1]), int(center[0])] = np.cos(a3)
                this_angle_map[5, int(center[1]), int(center[0])] = np.sin(a3)

            except:

                continue

        gt_hm.append(this_heat_map.cpu().numpy())
        gt_center_res.append(this_res_map)
        gt_center_dis_bin.append(this_dis_bin_map)
        gt_center_dis_res.append(this_dis_res_map)
        gt_dim.append(this_size_map)
        gt_dim2d.append(this_size_2d_map)
        gt_rot.append(this_angle_map)

    return np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dis_bin), np.array(gt_center_dis_res), np.array(gt_dim), np.array(gt_dim2d), np.array(gt_rot)

def center_point_encoder_DORN_cs(gt_box9d_with_cls,
                                 intrinsic_mat,
                                 extrinsic_mat,
                                 distortion_matrix,
                                 new_im_width=None,
                                 new_im_hight=None,
                                 raw_im_width=None,
                                 raw_im_hight=None,
                                 stride=None,
                                 im_num=None,
                                 class_name_config=None,
                                 center_rad=None,
                                 depth_num=None,
                                 max_dis=None,
                                 max_size=None):
    cls_num = len(class_name_config)

    gt_hm, gt_center_res, gt_center_dis_bin, gt_center_dis_res, gt_dim, gt_rot = [], [], [], [], [], []

    for im_id in range(im_num):
        in_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]

        xyz = copy.deepcopy(gt_box9d_with_cls[:, 0:3])

        ptx_im, depth = projectPoints(xyz, in_mat)  #
        ptx_im = ptx_im.reshape(-1, 2)
        depth = depth.reshape(-1)


        this_heat_map = torch.zeros(cls_num, new_im_hight // stride, new_im_width // stride)

        this_res_map = np.zeros(shape=(cls_num, 2, new_im_hight // stride, new_im_width // stride))

        this_dis_bin_map = np.zeros(shape=(cls_num, depth_num, new_im_hight // stride, new_im_width // stride))
        this_dis_res_map = np.zeros(shape=(cls_num, depth_num, new_im_hight // stride, new_im_width // stride))

        this_size_map = np.zeros(shape=(cls_num, 3, new_im_hight // stride, new_im_width // stride))

        this_angle_map = np.zeros(shape=(cls_num, 6, new_im_hight // stride, new_im_width // stride))

        for obj_i, obj in enumerate(gt_box9d_with_cls):

            this_cls = int(obj[-1])

            center = ptx_im[obj_i]

            dis = depth[obj_i]

            center[0] *= (new_im_width / raw_im_width / stride)
            center[1] *= (new_im_hight / raw_im_hight / stride)

            this_heat_map[this_cls] = draw_gaussian_to_heatmap(this_heat_map[this_cls], center, center_rad)



            this_res_map[this_cls][0], this_res_map[this_cls][1] = draw_res_to_heatmap(this_res_map[this_cls][0], this_res_map[this_cls][1], center)

            cls_aware_max_dis = max_dis[this_cls]
            cls_aware_max_size = max_size[this_cls]

            depth_interval = cls_aware_max_dis / depth_num


            try:

                dis_grid_id = int(dis / depth_interval)
                dis_residual = dis % depth_interval

                if dis_grid_id >= depth_num:
                    dis_grid_id = depth_num - 1
                    dis_residual = dis - (depth_num - 1) * depth_interval

                dis_residual = dis_residual / depth_interval

                this_dis_bin_map[this_cls, dis_grid_id, int(center[1]), int(center[0])] = 1

                this_dis_res_map[this_cls, dis_grid_id, int(center[1]), int(center[0])] = dis_residual

                l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]

                this_size_map[this_cls, 0, int(center[1]), int(center[0])] = l / cls_aware_max_size
                this_size_map[this_cls, 1, int(center[1]), int(center[0])] = w / cls_aware_max_size
                this_size_map[this_cls, 2, int(center[1]), int(center[0])] = h / cls_aware_max_size

                this_angle_map[this_cls, 0, int(center[1]), int(center[0])] = np.cos(a1)
                this_angle_map[this_cls, 1, int(center[1]), int(center[0])] = np.sin(a1)
                this_angle_map[this_cls, 2, int(center[1]), int(center[0])] = np.cos(a2)
                this_angle_map[this_cls, 3, int(center[1]), int(center[0])] = np.sin(a2)
                this_angle_map[this_cls, 4, int(center[1]), int(center[0])] = np.cos(a3)
                this_angle_map[this_cls, 5, int(center[1]), int(center[0])] = np.sin(a3)

            except:

                continue

        gt_hm.append(this_heat_map.cpu().numpy())
        gt_center_res.append(this_res_map)
        gt_center_dis_bin.append(this_dis_bin_map)
        gt_center_dis_res.append(this_dis_res_map)
        gt_dim.append(this_size_map)
        gt_rot.append(this_angle_map)

    return np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dis_bin), np.array(gt_center_dis_res), np.array(
        gt_dim), np.array(gt_rot)


def center_point_encoder_DORN_depth(gt_box9d_with_cls,
                           intrinsic_mat,
                           extrinsic_mat,
                           distortion_matrix,
                           new_im_width=None,
                           new_im_hight=None,
                           raw_im_width=None,
                           raw_im_hight=None,
                           stride=None,
                           im_num=None,
                           class_name_config=None,
                           center_rad=None,
                           depth_num = None,
                           depth_interval = None):


    cls_num = len(class_name_config)

    gt_hm, gt_center_res, gt_center_dep, gt_center_dis_bin, gt_center_dis_res, gt_dim, gt_rot = [], [], [], [], [], [], []

    for im_id in range(im_num):
        in_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]


        xyz = copy.deepcopy(gt_box9d_with_cls[:,0:3])

        ptx_im, depth = projectPoints(xyz, in_mat)#
        ptx_im = ptx_im.reshape(-1,2)
        depth = depth.reshape(-1,1)

        this_heat_map = torch.zeros(cls_num, new_im_hight // stride, new_im_width // stride)

        this_res_map = np.zeros(shape = (2, new_im_hight // stride, new_im_width // stride))

        this_dis_bin_map = np.zeros(shape = (depth_num, new_im_hight // stride, new_im_width // stride))
        this_dis_res_map = np.zeros(shape = (depth_num, new_im_hight // stride, new_im_width // stride))

        this_dis_dep_map = np.zeros(shape=(1, new_im_hight // stride, new_im_width // stride))

        this_size_map = np.zeros(shape = (3, new_im_hight // stride, new_im_width // stride))

        this_angle_map = np.zeros(shape = (6, new_im_hight // stride, new_im_width // stride))


        for obj_i, obj in enumerate(gt_box9d_with_cls):

            this_cls = int(obj[-1])

            center = ptx_im[obj_i]

            dis = depth[obj_i]

            center[0] *= (new_im_width / raw_im_width / stride)
            center[1] *= (new_im_hight / raw_im_hight / stride)

            this_heat_map[this_cls] = draw_gaussian_to_heatmap(this_heat_map[this_cls], center, center_rad)

            this_res_map[0], this_res_map[1] = draw_res_to_heatmap(this_res_map[0], this_res_map[1], center)

            try:

                dis_grid_id = int(dis/depth_interval)
                dis_residual = dis%depth_interval

                if dis_grid_id>=depth_num:
                    dis_grid_id = depth_num-1
                    dis_residual = dis-(depth_num-1)*depth_interval


                this_dis_bin_map[dis_grid_id, int(center[1]), int(center[0])] = 1

                this_dis_res_map[dis_grid_id, int(center[1]), int(center[0])] = dis_residual

                this_dis_dep_map[0, int(center[1]), int(center[0])] = dis

                l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]

                this_size_map[0, int(center[1]), int(center[0])] = l
                this_size_map[1, int(center[1]), int(center[0])] = w
                this_size_map[2, int(center[1]), int(center[0])] = h

                this_angle_map[0, int(center[1]), int(center[0])] = np.cos(a1)
                this_angle_map[1, int(center[1]), int(center[0])] = np.sin(a1)
                this_angle_map[2, int(center[1]), int(center[0])] = np.cos(a2)
                this_angle_map[3, int(center[1]), int(center[0])] = np.sin(a2)
                this_angle_map[4, int(center[1]), int(center[0])] = np.cos(a3)
                this_angle_map[5, int(center[1]), int(center[0])] = np.sin(a3)

            except:

                continue

        gt_hm.append(this_heat_map.cpu().numpy())
        gt_center_res.append(this_res_map)
        gt_center_dep.append(this_dis_dep_map)
        gt_center_dis_bin.append(this_dis_bin_map)
        gt_center_dis_res.append(this_dis_res_map)
        gt_dim.append(this_size_map)
        gt_rot.append(this_angle_map)


    return np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dep), np.array(gt_center_dis_bin), np.array(gt_center_dis_res), np.array(gt_dim), np.array(gt_rot)

def center_point_encoder_quat(gt_box9d_with_cls,
                               intrinsic_mat,
                               extrinsic_mat,
                               distortion_matrix,
                               new_im_width=None,
                               new_im_hight=None,
                               raw_im_width=None,
                               raw_im_hight=None,
                               stride=None,
                               im_num=None,
                               class_name_config=None,
                               center_rad=None):


        cls_num = len(class_name_config)

        gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = [], [], [], [], []

        for im_id in range(im_num):
            in_mat = intrinsic_mat[im_id]
            ex_mat = extrinsic_mat[im_id]
            dis_mat = distortion_matrix[im_id]


            xyz = copy.deepcopy(gt_box9d_with_cls[:,0:3])

            ptx_im, depth = projectPoints(xyz, in_mat)#
            ptx_im = ptx_im.reshape(-1,2)
            depth = depth.reshape(-1,1)

            this_heat_map = torch.zeros(cls_num, new_im_hight // stride, new_im_width // stride)

            this_res_map = np.zeros(shape = (2, new_im_hight // stride, new_im_width // stride))

            this_dis_map = np.zeros(shape = (1, new_im_hight // stride, new_im_width // stride))

            this_size_map = np.zeros(shape = (3, new_im_hight // stride, new_im_width // stride))

            this_angle_map = np.zeros(shape = (4, new_im_hight // stride, new_im_width // stride))


            for obj_i, obj in enumerate(gt_box9d_with_cls):

                this_cls = int(obj[-1])

                center = ptx_im[obj_i]

                dis = depth[obj_i]

                center[0] *= (new_im_width / raw_im_width / stride)
                center[1] *= (new_im_hight / raw_im_hight / stride)

                this_heat_map[this_cls] = draw_gaussian_to_heatmap(this_heat_map[this_cls], center, center_rad)

                this_res_map[0], this_res_map[1] = draw_res_to_heatmap(this_res_map[0], this_res_map[1], center)

                try:

                    this_dis_map[0, int(center[1]), int(center[0])] = dis

                    l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]

                    r_quat = R.from_euler('xyz', [a1, a2, a3], degrees=False)

                    r_quat = ( r_quat.as_quat()+1)/2

                    aa1, aa2, aa3, aa4 = r_quat[0], r_quat[1], r_quat[2], r_quat[3]

                    this_size_map[0, int(center[1]), int(center[0])] = l
                    this_size_map[1, int(center[1]), int(center[0])] = w
                    this_size_map[2, int(center[1]), int(center[0])] = h

                    this_angle_map[0, int(center[1]), int(center[0])] = aa1
                    this_angle_map[1, int(center[1]), int(center[0])] = aa2
                    this_angle_map[2, int(center[1]), int(center[0])] = aa3
                    this_angle_map[3, int(center[1]), int(center[0])] = aa4

                except:

                    continue

            gt_hm.append(this_heat_map.cpu().numpy())
            gt_center_res.append(this_res_map)
            gt_center_dis.append(this_dis_map)
            gt_dim.append(this_size_map)
            gt_rot.append(this_angle_map)

        return np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dis), np.array(gt_dim), np.array(gt_rot)


def center_point_encoder_rod(gt_box9d_with_cls,
                               intrinsic_mat,
                               extrinsic_mat,
                               distortion_matrix,
                               new_im_width=None,
                               new_im_hight=None,
                               raw_im_width=None,
                               raw_im_hight=None,
                               stride=None,
                               im_num=None,
                               class_name_config=None,
                               center_rad=None):


        cls_num = len(class_name_config)

        gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = [], [], [], [], []

        for im_id in range(im_num):
            in_mat = intrinsic_mat[im_id]
            ex_mat = extrinsic_mat[im_id]
            dis_mat = distortion_matrix[im_id]


            xyz = copy.deepcopy(gt_box9d_with_cls[:,0:3])

            ptx_im, depth = projectPoints(xyz, in_mat)#
            ptx_im = ptx_im.reshape(-1,2)
            depth = depth.reshape(-1,1)

            this_heat_map = torch.zeros(cls_num, new_im_hight // stride, new_im_width // stride)

            this_res_map = np.zeros(shape = (2, new_im_hight // stride, new_im_width // stride))

            this_dis_map = np.zeros(shape = (1, new_im_hight // stride, new_im_width // stride))

            this_size_map = np.zeros(shape = (3, new_im_hight // stride, new_im_width // stride))

            this_angle_map = np.zeros(shape = (6, new_im_hight // stride, new_im_width // stride))


            for obj_i, obj in enumerate(gt_box9d_with_cls):

                this_cls = int(obj[-1])

                center = ptx_im[obj_i]

                dis = depth[obj_i]

                center[0] *= (new_im_width / raw_im_width / stride)
                center[1] *= (new_im_hight / raw_im_hight / stride)

                this_heat_map[this_cls] = draw_gaussian_to_heatmap(this_heat_map[this_cls], center, center_rad)

                this_res_map[0], this_res_map[1] = draw_res_to_heatmap(this_res_map[0], this_res_map[1], center)

                try:

                    this_dis_map[0, int(center[1]), int(center[0])] = dis

                    l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]

                    angle_rod = R.from_euler('xyz', np.array([[a1, a2, a3]]), False)
                    angle_rod = angle_rod.as_rotvec()
                    a1, a2, a3 = angle_rod[0,0], angle_rod[0,0], angle_rod[0,0]


                    this_size_map[0, int(center[1]), int(center[0])] = l
                    this_size_map[1, int(center[1]), int(center[0])] = w
                    this_size_map[2, int(center[1]), int(center[0])] = h

                    this_angle_map[0, int(center[1]), int(center[0])] = np.cos(a1)
                    this_angle_map[1, int(center[1]), int(center[0])] = np.sin(a1)
                    this_angle_map[2, int(center[1]), int(center[0])] = np.cos(a2)
                    this_angle_map[3, int(center[1]), int(center[0])] = np.sin(a2)
                    this_angle_map[4, int(center[1]), int(center[0])] = np.cos(a3)
                    this_angle_map[5, int(center[1]), int(center[0])] = np.sin(a3)

                except:

                    continue

            gt_hm.append(this_heat_map.cpu().numpy())
            gt_center_res.append(this_res_map)
            gt_center_dis.append(this_dis_map)
            gt_dim.append(this_size_map)
            gt_rot.append(this_angle_map)

        return np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dis), np.array(gt_dim), np.array(gt_rot)


def nms_pytorch(confidence_map, max_num, distance_threshold=5):
    """
    PyTorch implementation of Non-Maximum Suppression

    Args:
        confidence_map (torch.Tensor): 2D confidence map with shape (H, W)
        max_num (int): Maximum number of points to return
        distance_threshold (float): Pixel distance threshold for suppression (default: 5)

    Returns:
        tuple: (confidences, linear_indices)
            - confidences: Selected confidence values, shape (k,) where k <= max_num
            - linear_indices: Linear indices of selected points in flattened array, shape (k,)
    """
    device = confidence_map.device
    H, W = confidence_map.shape

    # Flatten the confidence map
    flat_conf = confidence_map.flatten()

    # Get initial top-k candidates
    initial_k = min(max_num , flat_conf.numel())  # Get more candidates initially
    top_conf, top_indices = torch.topk(flat_conf, k=initial_k)

    # Convert linear indices to 2D coordinates
    top_y = top_indices // W
    top_x = top_indices % W
    top_coords = torch.stack([top_x, top_y], dim=1).float()  # Shape: (initial_k, 2)

    # NMS process
    selected_mask = torch.ones(len(top_indices), dtype=torch.bool, device=device)

    for i in range(len(top_indices)):
        if not selected_mask[i]:
            continue

        # Current point coordinates
        current_coord = top_coords[i:i + 1]  # Shape: (1, 2)

        # Calculate distances to all remaining points
        remaining_coords = top_coords[i + 1:]  # Shape: (remaining, 2)
        if len(remaining_coords) == 0:
            break

        # Euclidean distance
        distances = torch.norm(remaining_coords - current_coord, dim=1)  # Shape: (remaining,)

        # Suppress points within distance threshold
        suppress_mask = distances < distance_threshold
        selected_mask[i + 1:] = selected_mask[i + 1:] & (~suppress_mask)

    # Get final selected results
    selected_indices = torch.where(selected_mask)[0][:max_num]

    # Return results in the requested format
    confi = top_conf[selected_indices]
    linear_indices = top_indices[selected_indices]

    return confi, linear_indices

def center_point_decoder(hm,
                        center_res,
                        center_dis,
                        dim,
                        rot,
                        intrinsic_mat,
                        extrinsic_mat,
                        distortion_matrix,
                        new_im_width=None,
                        new_im_hight=None,
                        raw_im_width=None,
                        raw_im_hight=None,
                        stride=None,
                        im_num=None,
                        max_num = 10):



    pred_boxes9d = []

    all_confidence = []

    for im_id in range(im_num):

        im_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]

        this_hm = hm[im_id]  # c, W, H
        this_hm_conf, cls_map = this_hm.max(0)

        this_center_res = center_res[im_id]
        this_center_dis = center_dis[im_id]
        this_dim = dim[im_id]
        this_rot = rot[im_id]

        shape_hm = this_hm_conf.shape

        # Call NMS instead of simple topk

        confi, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)  # (N,) (N,)

        confi = confi.detach().cpu().numpy()

        c_num = shape_hm[-1]
        rows = linear_indices // c_num  # K
        cols = linear_indices % c_num  # K


        rows_ind = rows.long()
        cols_ind = cols.long()

        res_x = this_center_res[0, rows_ind, cols_ind]
        res_y = this_center_res[1, rows_ind, cols_ind]

        cls = cls_map[rows_ind, cols_ind].cpu().numpy()
        cls = cls.reshape(-1,1)


        depth = this_center_dis[0, rows_ind, cols_ind].detach().cpu().numpy()
        y_cor = rows+res_y
        x_cor = cols+res_x
        y_cor = y_cor.detach().cpu().numpy()
        x_cor = x_cor.detach().cpu().numpy()

        delta0 = (new_im_width / raw_im_width / stride)
        delta1 = (new_im_hight / raw_im_hight / stride)

        x_cor /= delta0
        y_cor /= delta1

        x_cor = x_cor.reshape(-1,1)
        y_cor = y_cor.reshape(-1,1)


        points = backProject(np.concatenate([x_cor, y_cor],-1), depth, im_mat)

        l, w, h = this_dim[0, rows_ind, cols_ind], this_dim[1, rows_ind, cols_ind], this_dim[2, rows_ind, cols_ind]

        a1 = torch.atan2(this_rot[1, rows_ind, cols_ind], this_rot[0, rows_ind, cols_ind])
        a2 = torch.atan2(this_rot[3, rows_ind, cols_ind], this_rot[2, rows_ind, cols_ind])
        a3 = torch.atan2(this_rot[5, rows_ind, cols_ind], this_rot[4, rows_ind, cols_ind])

        l = l.detach().cpu().numpy().reshape(-1, 1)
        w = w.detach().cpu().numpy().reshape(-1, 1)
        h = h.detach().cpu().numpy().reshape(-1, 1)
        a1 = a1.detach().cpu().numpy().reshape(-1, 1)
        a2 = a2.detach().cpu().numpy().reshape(-1, 1)
        a3 = a3.detach().cpu().numpy().reshape(-1, 1)


        this_box9d = np.concatenate([points,l,w,h,a1,a2,a3,cls], -1)

        pred_boxes9d.append(this_box9d)
        all_confidence.append(confi)

    return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)



def center_point_decoder_DORN(hm,
                        center_res,
                        center_dis_bin,
                        center_dis_res,
                        dim,
                        rot,
                        intrinsic_mat,
                        extrinsic_mat,
                        distortion_matrix,
                        new_im_width=None,
                        new_im_hight=None,
                        raw_im_width=None,
                        raw_im_hight=None,
                        stride=None,
                        im_num=None,
                        max_num = 10,
                        depth_interval=10):



    pred_boxes9d = []

    all_confidence = []

    for im_id in range(im_num):

        im_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]

        this_hm = hm[im_id]  # c, W, H
        this_hm_conf, cls_map = this_hm.max(0)

        this_center_res = center_res[im_id]
        this_center_dis_bin = center_dis_bin[im_id]
        this_center_dis_res = center_dis_res[im_id]
        this_dim = dim[im_id]
        this_rot = rot[im_id]

        shape_hm = this_hm_conf.shape

        # Call NMS instead of simple topk

        confi, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)  # (N,) (N,)

        confi = confi.detach().cpu().numpy()

        c_num = shape_hm[-1]
        rows = linear_indices // c_num  # K
        cols = linear_indices % c_num  # K


        rows_ind = rows.long()
        cols_ind = cols.long()

        res_x = this_center_res[0, rows_ind, cols_ind]
        res_y = this_center_res[1, rows_ind, cols_ind]

        cls = cls_map[rows_ind, cols_ind].cpu().numpy()
        cls = cls.reshape(-1,1)


        depth_bin = this_center_dis_bin[:, rows_ind, cols_ind] # 60, 55

        depth_res = this_center_dis_res[:, rows_ind, cols_ind] # 60, 55

        max_value, _ = depth_bin.max(dim=0) # 55

        argmax_id = depth_bin.argmax(dim=0) # 55

        max_res = depth_res[argmax_id, torch.arange(argmax_id.size(0))]

        depth = depth_interval*argmax_id+max_res*depth_interval

        depth = depth.detach().cpu().numpy()

        y_cor = rows+res_y
        x_cor = cols+res_x
        y_cor = y_cor.detach().cpu().numpy()
        x_cor = x_cor.detach().cpu().numpy()

        delta0 = (new_im_width / raw_im_width / stride)
        delta1 = (new_im_hight / raw_im_hight / stride)

        x_cor /= delta0
        y_cor /= delta1

        x_cor = x_cor.reshape(-1,1)
        y_cor = y_cor.reshape(-1,1)


        points = backProject(np.concatenate([x_cor, y_cor],-1), depth, im_mat)

        l, w, h = this_dim[0, rows_ind, cols_ind], this_dim[1, rows_ind, cols_ind], this_dim[2, rows_ind, cols_ind]

        a1 = torch.atan2(this_rot[1, rows_ind, cols_ind], this_rot[0, rows_ind, cols_ind])
        a2 = torch.atan2(this_rot[3, rows_ind, cols_ind], this_rot[2, rows_ind, cols_ind])
        a3 = torch.atan2(this_rot[5, rows_ind, cols_ind], this_rot[4, rows_ind, cols_ind])

        l = l.detach().cpu().numpy().reshape(-1, 1)
        w = w.detach().cpu().numpy().reshape(-1, 1)
        h = h.detach().cpu().numpy().reshape(-1, 1)
        a1 = a1.detach().cpu().numpy().reshape(-1, 1)
        a2 = a2.detach().cpu().numpy().reshape(-1, 1)
        a3 = a3.detach().cpu().numpy().reshape(-1, 1)


        this_box9d = np.concatenate([points,l,w,h,a1,a2,a3,cls], -1)


        pred_boxes9d.append(this_box9d)
        all_confidence.append(confi)

    return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)


def center_point_decoder_DORN_focus_640(hm,
                        center_res,
                        center_dis_bin,
                        center_dis_res,
                        dim,
                        rot,
                        intrinsic_mat,
                        extrinsic_mat,
                        distortion_matrix,
                        new_im_width=None,
                        new_im_hight=None,
                        raw_im_width=None,
                        raw_im_hight=None,
                        stride=None,
                        im_num=None,
                        max_num = 10,
                        depth_interval=10):



    pred_boxes9d = []

    all_confidence = []

    for im_id in range(im_num):

        im_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]

        defult_focus = 640

        focus_scale = im_mat[0, 0] / defult_focus

        new_im_mat= copy.deepcopy(im_mat)

        new_im_mat[0:2,0:2] = im_mat[0:2,0:2] / focus_scale

        this_hm = hm[im_id]  # c, W, H
        this_hm_conf, cls_map = this_hm.max(0)

        this_center_res = center_res[im_id]
        this_center_dis_bin = center_dis_bin[im_id]
        this_center_dis_res = center_dis_res[im_id]
        this_dim = dim[im_id]
        this_rot = rot[im_id]

        shape_hm = this_hm_conf.shape

        # Call NMS instead of simple topk

        confi, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)  # (N,) (N,)

        confi = confi.detach().cpu().numpy()

        c_num = shape_hm[-1]
        rows = linear_indices // c_num  # K
        cols = linear_indices % c_num  # K


        rows_ind = rows.long()
        cols_ind = cols.long()

        res_x = this_center_res[0, rows_ind, cols_ind]
        res_y = this_center_res[1, rows_ind, cols_ind]

        cls = cls_map[rows_ind, cols_ind].cpu().numpy()
        cls = cls.reshape(-1,1)


        depth_bin = this_center_dis_bin[:, rows_ind, cols_ind] # 60, 55

        depth_res = this_center_dis_res[:, rows_ind, cols_ind] # 60, 55

        max_value, _ = depth_bin.max(dim=0) # 55

        argmax_id = depth_bin.argmax(dim=0) # 55

        max_res = depth_res[argmax_id, torch.arange(argmax_id.size(0))]

        depth = depth_interval*argmax_id+max_res*depth_interval

        depth = depth.detach().cpu().numpy()

        y_cor = rows+res_y
        x_cor = cols+res_x
        y_cor = y_cor.detach().cpu().numpy()
        x_cor = x_cor.detach().cpu().numpy()

        delta0 = (new_im_width / raw_im_width / stride)
        delta1 = (new_im_hight / raw_im_hight / stride)

        x_cor /= delta0
        y_cor /= delta1

        x_cor = x_cor.reshape(-1,1)
        y_cor = y_cor.reshape(-1,1)


        points = backProject(np.concatenate([x_cor, y_cor],-1), depth, new_im_mat)

        l, w, h = this_dim[0, rows_ind, cols_ind], this_dim[1, rows_ind, cols_ind], this_dim[2, rows_ind, cols_ind]

        a1 = torch.atan2(this_rot[1, rows_ind, cols_ind], this_rot[0, rows_ind, cols_ind])
        a2 = torch.atan2(this_rot[3, rows_ind, cols_ind], this_rot[2, rows_ind, cols_ind])
        a3 = torch.atan2(this_rot[5, rows_ind, cols_ind], this_rot[4, rows_ind, cols_ind])

        l = l.detach().cpu().numpy().reshape(-1, 1)
        w = w.detach().cpu().numpy().reshape(-1, 1)
        h = h.detach().cpu().numpy().reshape(-1, 1)
        a1 = a1.detach().cpu().numpy().reshape(-1, 1)
        a2 = a2.detach().cpu().numpy().reshape(-1, 1)
        a3 = a3.detach().cpu().numpy().reshape(-1, 1)

        points[:, 2] *= focus_scale

        this_box9d = np.concatenate([points,l,w,h,a1,a2,a3,cls], -1)

        pred_boxes9d.append(this_box9d)
        all_confidence.append(confi)

    return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)

def center_point_decoder_DORN_focus_1280(hm,
                        center_res,
                        center_dis_bin,
                        center_dis_res,
                        dim,
                        rot,
                        intrinsic_mat,
                        extrinsic_mat,
                        distortion_matrix,
                        new_im_width=None,
                        new_im_hight=None,
                        raw_im_width=None,
                        raw_im_hight=None,
                        stride=None,
                        im_num=None,
                        max_num = 10,
                        depth_interval=10):



    pred_boxes9d = []

    all_confidence = []

    for im_id in range(im_num):

        im_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]

        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]

        defult_focus = 1280

        focus_scale = im_mat[0, 0] / defult_focus

        new_im_mat = copy.deepcopy(im_mat)

        new_im_mat[0:2, 0:2] = im_mat[0:2, 0:2] / focus_scale

        this_hm = hm[im_id]  # c, W, H
        this_hm_conf, cls_map = this_hm.max(0)

        this_center_res = center_res[im_id]
        this_center_dis_bin = center_dis_bin[im_id]
        this_center_dis_res = center_dis_res[im_id]
        this_dim = dim[im_id]
        this_rot = rot[im_id]

        shape_hm = this_hm_conf.shape

        # Call NMS instead of simple topk

        confi, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)  # (N,) (N,)

        confi = confi.detach().cpu().numpy()

        c_num = shape_hm[-1]
        rows = linear_indices // c_num  # K
        cols = linear_indices % c_num  # K


        rows_ind = rows.long()
        cols_ind = cols.long()

        res_x = this_center_res[0, rows_ind, cols_ind]
        res_y = this_center_res[1, rows_ind, cols_ind]

        cls = cls_map[rows_ind, cols_ind].cpu().numpy()
        cls = cls.reshape(-1,1)


        depth_bin = this_center_dis_bin[:, rows_ind, cols_ind] # 60, 55

        depth_res = this_center_dis_res[:, rows_ind, cols_ind] # 60, 55

        max_value, _ = depth_bin.max(dim=0) # 55

        argmax_id = depth_bin.argmax(dim=0) # 55

        max_res = depth_res[argmax_id, torch.arange(argmax_id.size(0))]

        depth = depth_interval*argmax_id+max_res*depth_interval

        depth = depth.detach().cpu().numpy()

        y_cor = rows+res_y
        x_cor = cols+res_x
        y_cor = y_cor.detach().cpu().numpy()
        x_cor = x_cor.detach().cpu().numpy()

        delta0 = (new_im_width / raw_im_width / stride)
        delta1 = (new_im_hight / raw_im_hight / stride)

        x_cor /= delta0
        y_cor /= delta1

        x_cor = x_cor.reshape(-1,1)
        y_cor = y_cor.reshape(-1,1)


        points = backProject(np.concatenate([x_cor, y_cor],-1), depth, new_im_mat)

        l, w, h = this_dim[0, rows_ind, cols_ind], this_dim[1, rows_ind, cols_ind], this_dim[2, rows_ind, cols_ind]

        a1 = torch.atan2(this_rot[1, rows_ind, cols_ind], this_rot[0, rows_ind, cols_ind])
        a2 = torch.atan2(this_rot[3, rows_ind, cols_ind], this_rot[2, rows_ind, cols_ind])
        a3 = torch.atan2(this_rot[5, rows_ind, cols_ind], this_rot[4, rows_ind, cols_ind])

        l = l.detach().cpu().numpy().reshape(-1, 1)
        w = w.detach().cpu().numpy().reshape(-1, 1)
        h = h.detach().cpu().numpy().reshape(-1, 1)
        a1 = a1.detach().cpu().numpy().reshape(-1, 1)
        a2 = a2.detach().cpu().numpy().reshape(-1, 1)
        a3 = a3.detach().cpu().numpy().reshape(-1, 1)


        points[:, 2] *= focus_scale


        this_box9d = np.concatenate([points,l,w,h,a1,a2,a3,cls], -1)

        pred_boxes9d.append(this_box9d)
        all_confidence.append(confi)

    return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)

def center_point_decoder_DORN_cs(hm,
                        center_res,
                        center_dis_bin,
                        center_dis_res,
                        dim,
                        rot,
                        intrinsic_mat,
                        extrinsic_mat,
                        distortion_matrix,
                        new_im_width=None,
                        new_im_hight=None,
                        raw_im_width=None,
                        raw_im_hight=None,
                        stride=None,
                        im_num=None,
                        max_num = 10,
                        depth_num=None,
                        max_dis=None,
                        max_size=None):



    pred_boxes9d = []

    all_confidence = []

    for im_id in range(im_num):

        im_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]

        this_hm = hm[im_id]  # c, W, H

        this_center_res = center_res[im_id] # cls, C, W, H
        this_center_dis_bin = center_dis_bin[im_id] # cls, C, W, H
        this_center_dis_res = center_dis_res[im_id] # cls, C, W, H
        this_dim = dim[im_id] # cls, C, W, H
        this_rot = rot[im_id] # cls, C, W, H

        cls_num, C, W, H  = this_rot.shape

        for cls_id in range(cls_num):
            this_cls_center_res = this_center_res[cls_id]  # C, W, H
            this_cls_center_dis_bin = this_center_dis_bin[cls_id]  # C, W, H
            this_cls_center_dis_res = this_center_dis_res[cls_id]  # C, W, H
            this_cls_dim = this_dim[cls_id]  # C, W, H
            this_cls_rot = this_rot[cls_id]  # C, W, H


            this_hm_conf = this_hm[cls_id]

            shape_hm = this_hm_conf.shape

            # Call NMS instead of simple topk

            confi, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)  # (N,) (N,)



            confi = confi.detach().cpu().numpy()

            c_num = shape_hm[-1]
            rows = linear_indices // c_num  # K
            cols = linear_indices % c_num  # K

            rows_ind = rows.long()
            cols_ind = cols.long()

            res_x = this_cls_center_res[0, rows_ind, cols_ind]
            res_y = this_cls_center_res[1, rows_ind, cols_ind]

            cls = np.ones(shape=res_y.shape)*cls_id


            cls = cls.reshape(-1,1)

            depth_bin = this_cls_center_dis_bin[:, rows_ind, cols_ind] # 60, 55

            depth_res = this_cls_center_dis_res[:, rows_ind, cols_ind] # 60, 55

            max_value, _ = depth_bin.max(dim=0) # 55

            argmax_id = depth_bin.argmax(dim=0) # 55

            max_res = depth_res[argmax_id, torch.arange(argmax_id.size(0))]

            depth_interval = max_dis[cls_id]/depth_num



            depth = depth_interval*argmax_id+max_res*depth_interval

            depth = depth.detach().cpu().numpy()

            y_cor = rows+res_y
            x_cor = cols+res_x
            y_cor = y_cor.detach().cpu().numpy()
            x_cor = x_cor.detach().cpu().numpy()

            delta0 = (new_im_width / raw_im_width / stride)
            delta1 = (new_im_hight / raw_im_hight / stride)

            x_cor /= delta0
            y_cor /= delta1

            x_cor = x_cor.reshape(-1,1)
            y_cor = y_cor.reshape(-1,1)


            points = backProject(np.concatenate([x_cor, y_cor],-1), depth, im_mat)

            l, w, h = this_cls_dim[0, rows_ind, cols_ind], this_cls_dim[1, rows_ind, cols_ind], this_cls_dim[2, rows_ind, cols_ind]

            a1 = torch.atan2(this_cls_rot[1, rows_ind, cols_ind], this_cls_rot[0, rows_ind, cols_ind])
            a2 = torch.atan2(this_cls_rot[3, rows_ind, cols_ind], this_cls_rot[2, rows_ind, cols_ind])
            a3 = torch.atan2(this_cls_rot[5, rows_ind, cols_ind], this_cls_rot[4, rows_ind, cols_ind])

            l = l.detach().cpu().numpy().reshape(-1, 1)
            w = w.detach().cpu().numpy().reshape(-1, 1)
            h = h.detach().cpu().numpy().reshape(-1, 1)

            this_max_size = max_size[cls_id]


            l = l*this_max_size
            w = w*this_max_size
            h = h*this_max_size

            a1 = a1.detach().cpu().numpy().reshape(-1, 1)
            a2 = a2.detach().cpu().numpy().reshape(-1, 1)
            a3 = a3.detach().cpu().numpy().reshape(-1, 1)


            this_box9d = np.concatenate([points,l,w,h,a1,a2,a3,cls], -1)

            pred_boxes9d.append(this_box9d)
            all_confidence.append(confi)


    out_confi = np.concatenate(all_confidence)
    out_box = np.concatenate(pred_boxes9d)


    return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)


def center_point_decoder_quat(hm,
                            center_res,
                            center_dis,
                            dim,
                            rot,
                            intrinsic_mat,
                            extrinsic_mat,
                            distortion_matrix,
                            new_im_width=None,
                            new_im_hight=None,
                            raw_im_width=None,
                            raw_im_hight=None,
                            stride=None,
                            im_num=None,
                            max_num = 10):



        pred_boxes9d = []

        all_confidence = []

        for im_id in range(im_num):

            im_mat = intrinsic_mat[im_id]
            ex_mat = extrinsic_mat[im_id]
            dis_mat = distortion_matrix[im_id]

            this_hm = hm[im_id]  # c, W, H
            this_hm_conf, cls_map = this_hm.max(0)

            this_center_res = center_res[im_id]
            this_center_dis = center_dis[im_id]
            this_dim = dim[im_id]
            this_rot = rot[im_id]

            shape_hm = this_hm_conf.shape

            # Call NMS instead of simple topk

            confi, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)  # (N,) (N,)

            confi = confi.detach().cpu().numpy()

            c_num = shape_hm[-1]
            rows = linear_indices // c_num  # K
            cols = linear_indices % c_num  # K


            rows_ind = rows.long()
            cols_ind = cols.long()

            res_x = this_center_res[0, rows_ind, cols_ind]
            res_y = this_center_res[1, rows_ind, cols_ind]

            cls = cls_map[rows_ind, cols_ind].cpu().numpy()
            cls = cls.reshape(-1,1)


            depth = this_center_dis[0, rows_ind, cols_ind].detach().cpu().numpy()
            y_cor = rows+res_y
            x_cor = cols+res_x
            y_cor = y_cor.detach().cpu().numpy()
            x_cor = x_cor.detach().cpu().numpy()

            delta0 = (new_im_width / raw_im_width / stride)
            delta1 = (new_im_hight / raw_im_hight / stride)

            x_cor /= delta0
            y_cor /= delta1

            x_cor = x_cor.reshape(-1,1)
            y_cor = y_cor.reshape(-1,1)


            points = backProject(np.concatenate([x_cor, y_cor],-1), depth, im_mat)

            l, w, h = this_dim[0, rows_ind, cols_ind], this_dim[1, rows_ind, cols_ind], this_dim[2, rows_ind, cols_ind]

            aa1 = this_rot[0, rows_ind, cols_ind]
            aa2 = this_rot[1, rows_ind, cols_ind]
            aa3 = this_rot[2, rows_ind, cols_ind]
            aa4 = this_rot[3, rows_ind, cols_ind]


            l = l.detach().cpu().numpy().reshape(-1, 1)
            w = w.detach().cpu().numpy().reshape(-1, 1)
            h = h.detach().cpu().numpy().reshape(-1, 1)
            aa1 = aa1.detach().cpu().numpy().reshape(-1, 1)
            aa2 = aa2.detach().cpu().numpy().reshape(-1, 1)
            aa3 = aa3.detach().cpu().numpy().reshape(-1, 1)
            aa4 = aa4.detach().cpu().numpy().reshape(-1, 1)

            a_quat = np.concatenate([aa1, aa2, aa3, aa4], axis=-1)*2-1
            rotations = R.from_quat(a_quat)
            euler_angles = rotations.as_euler('xyz', degrees=False)

            this_box9d = np.concatenate([points,l,w,h,euler_angles,cls], -1)

            pred_boxes9d.append(this_box9d)
            all_confidence.append(confi)

        return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)


def center_point_decoder_rod(hm,
                            center_res,
                            center_dis,
                            dim,
                            rot,
                            intrinsic_mat,
                            extrinsic_mat,
                            distortion_matrix,
                            new_im_width=None,
                            new_im_hight=None,
                            raw_im_width=None,
                            raw_im_hight=None,
                            stride=None,
                            im_num=None,
                            max_num = 10):

        pred_boxes9d = []

        all_confidence = []

        for im_id in range(im_num):

            im_mat = intrinsic_mat[im_id]
            ex_mat = extrinsic_mat[im_id]
            dis_mat = distortion_matrix[im_id]

            this_hm = hm[im_id]  # c, W, H
            this_hm_conf, cls_map = this_hm.max(0)

            this_center_res = center_res[im_id]
            this_center_dis = center_dis[im_id]
            this_dim = dim[im_id]
            this_rot = rot[im_id]

            shape_hm = this_hm_conf.shape

            # Call NMS instead of simple topk

            confi, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)  # (N,) (N,)

            confi = confi.detach().cpu().numpy()

            c_num = shape_hm[-1]
            rows = linear_indices // c_num  # K
            cols = linear_indices % c_num  # K


            rows_ind = rows.long()
            cols_ind = cols.long()

            res_x = this_center_res[0, rows_ind, cols_ind]
            res_y = this_center_res[1, rows_ind, cols_ind]

            cls = cls_map[rows_ind, cols_ind].cpu().numpy()
            cls = cls.reshape(-1, 1)


            depth = this_center_dis[0, rows_ind, cols_ind].detach().cpu().numpy()
            y_cor = rows+res_y
            x_cor = cols+res_x
            y_cor = y_cor.detach().cpu().numpy()
            x_cor = x_cor.detach().cpu().numpy()

            delta0 = (new_im_width / raw_im_width / stride)
            delta1 = (new_im_hight / raw_im_hight / stride)

            x_cor /= delta0
            y_cor /= delta1

            x_cor = x_cor.reshape(-1,1)
            y_cor = y_cor.reshape(-1,1)


            points = backProject(np.concatenate([x_cor, y_cor],-1), depth, im_mat)

            l, w, h = this_dim[0, rows_ind, cols_ind], this_dim[1, rows_ind, cols_ind], this_dim[2, rows_ind, cols_ind]

            a1 = torch.atan2(this_rot[1, rows_ind, cols_ind], this_rot[0, rows_ind, cols_ind])
            a2 = torch.atan2(this_rot[3, rows_ind, cols_ind], this_rot[2, rows_ind, cols_ind])
            a3 = torch.atan2(this_rot[5, rows_ind, cols_ind], this_rot[4, rows_ind, cols_ind])

            l = l.detach().cpu().numpy().reshape(-1, 1)
            w = w.detach().cpu().numpy().reshape(-1, 1)
            h = h.detach().cpu().numpy().reshape(-1, 1)
            a1 = a1.detach().cpu().numpy().reshape(-1, 1)
            a2 = a2.detach().cpu().numpy().reshape(-1, 1)
            a3 = a3.detach().cpu().numpy().reshape(-1, 1)

            angle_rod = np.concatenate([a1, a2, a3], axis=-1)

            angle_rod = R.from_rotvec(angle_rod)
            angle_elu = angle_rod.as_euler('xyz', False)

            this_box9d = np.concatenate([points, l, w, h, angle_elu, cls], -1)

            pred_boxes9d.append(this_box9d)
            all_confidence.append(confi)

        return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)



all_object_encoders = {'key_point_encoder': key_point_encoder,
                       'key_point_decoder': key_point_decoder,
                       'center_point_encoder':center_point_encoder,
                       'center_point_decoder':center_point_decoder,
                       'center_point_encoder_DORN_depth':center_point_encoder_DORN_depth,
                       'center_point_encoder_DORN': center_point_encoder_DORN,
                       'center_point_decoder_DORN': center_point_decoder_DORN,
                       'center_point_encoder_DORN_w2d':center_point_encoder_DORN_w2d,
                       'center_point_encoder_DORN_cs': center_point_encoder_DORN_cs,
                       'center_point_decoder_DORN_cs': center_point_decoder_DORN_cs,
                       'center_point_encoder_quat': center_point_encoder_quat,
                       'center_point_decoder_quat': center_point_decoder_quat,
                       'center_point_encoder_rod': center_point_encoder_rod,
                       'center_point_decoder_rod': center_point_decoder_rod,
                       'center_point_encoder_DORN_focus_640': center_point_encoder_DORN_focus_640,
                       'center_point_decoder_DORN_focus_640': center_point_decoder_DORN_focus_640,
                       'center_point_encoder_DORN_focus_1280': center_point_encoder_DORN_focus_1280,
                       'center_point_decoder_DORN_focus_1280': center_point_decoder_DORN_focus_1280,
                       }

