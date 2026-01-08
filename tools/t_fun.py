import torch
import torch.nn as nn
import pickle as pkl
import os
import torch.optim as optim
import numpy as np
import copy

def projectPoints(corners, rotation_mat, translation_vec, intrinsic_mat, distortion_matrix):

    corners_homogeneous = np.hstack((corners, np.ones((corners.shape[0], 1))))

    extrinsic_mat = np.hstack((rotation_mat, translation_vec.reshape(3, 1)))

    corners_cam = np.dot(extrinsic_mat, corners_homogeneous.T).T

    depth = corners_cam[:, 2]

    corners_norm = corners_cam / corners_cam[:, 2].reshape(-1, 1)

    x = corners_norm[:, 0]
    y = corners_norm[:, 1]

    r2 = x**2 + y**2

    k1, k2, p1, p2, k3 = distortion_matrix
    radial_distortion = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_distorted = x * radial_distortion + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_distorted = y * radial_distortion + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    corners_2d = np.dot(intrinsic_mat, np.vstack((x_distorted, y_distorted, np.ones_like(x_distorted)))).T[:, :2]

    return corners_2d, depth

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_units=[128, 64, 16], output_dim=1,
                 activation='relu', dropout_rate=0.0):
        super(FlexibleMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for i, units in enumerate(hidden_units):
            layers.append(nn.Linear(prev_dim, units))

            # 选择激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.SiLU())

            layers.append(nn.BatchNorm1d(units))

            prev_dim = units

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

config = {
    'hidden_units': [2,2],  # 三层神经元数量
    'activation': 'relu',
    'dropout_rate': 0.0,
    'learning_rate': 0.0001
}

# 创建模型
model = FlexibleMLP(
    input_dim=2,
    hidden_units=config['hidden_units'],
    output_dim=1,
    activation=config['activation'],
    dropout_rate=config['dropout_rate']
)
model.cuda()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 用于回归任务
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

path_root = '/userhome/wuhai/LAA3D'


infos_train_path = os.path.join(path_root,'train4_info.pkl')
infos_val_path = os.path.join(path_root,'val4_info.pkl')


with open(infos_train_path, 'rb') as f:
    infos_train = pkl.load(f)

with open(infos_val_path, 'rb') as f:
    infos_val = pkl.load(f)


# 2+3 -> 5, 10

def generate_dataset(infos, batch_size = 80):

    all_x_batch = [] # N, 80, 5
    all_y_batch = [] # N, 80, 1

    each_x = [] # 80, 5
    each_y = [] # 80, 1

    interval = 1

    randoms = np.random.permutation(100000)

    for k in randoms: #len(infos)

        info = infos[int(k)]

        d2_dim = 0
        d3_dim = 0

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

        DiffLevel = 6

        mask1 = this_fine_class=='drone_unk2'
        maskl2 = diff <DiffLevel

        mask = maskl2 #mask1*

        boxes9d = boxes9d[mask]
        this_coarse_class = this_coarse_class[mask]
        this_boxes_2d = this_boxes_2d[mask]
        diff = diff[mask]

        if len(boxes9d)<1:
            continue

        image_info = info['image_info']
        intrinsic = image_info['intrinsic']
        extrinsic = image_info['extrinsic']

        xyz = copy.deepcopy(boxes9d[:, 0:3])

        ptx_im, depth = projectPoints(xyz, extrinsic[:3, :3], extrinsic[:3, 3], intrinsic, np.array([0, 0, 0, 0, 0]) * 0)  #
        ptx_im = ptx_im.reshape(-1, 2)
        depth = depth.reshape(-1, 1)

        for each_i, box9d in enumerate(boxes9d):

            box2d = this_boxes_2d[each_i]

            each_depth = depth[each_i]

            x_2d = box2d[2]-box2d[0]
            y_2d = box2d[3] - box2d[1]

            if len(each_x) < batch_size:
                each_x.append([box9d[5]/x_2d, box9d[3]/y_2d] ) #1./(box2d[3]-box2d[1]),  box9d[3], box9d[4], box9d[5]]

                each_y.append([each_depth[0]])

                # print([box2d[2]-box2d[0], box2d[3]-box2d[1], box9d[3], box9d[4], box9d[5]])
                # print([each_depth])
                # input()
            else:
                all_x_batch.append(copy.deepcopy(each_x))
                all_y_batch.append(copy.deepcopy(each_y))
                each_x = []
                each_y = []

    return all_x_batch, all_y_batch


all_x_batch, all_y_batch = generate_dataset(infos_train)

for e in range(1000):
    for i, each_x in enumerate(all_x_batch):
        each_y = all_y_batch[i]

        each_x = torch.from_numpy(np.array(each_x).astype(np.float32)).cuda()
        each_y = torch.from_numpy(np.array(each_y).astype(np.float32)).cuda()

        optimizer.zero_grad()

        pred_y = model(each_x)

        print(each_y.shape)
        print(pred_y.shape)

        loss_all = torch.abs(each_y-pred_y)

        print('max',loss_all.max())

        loss = loss_all.mean()

        print('loss',loss)

        loss.backward()

        optimizer.step()
