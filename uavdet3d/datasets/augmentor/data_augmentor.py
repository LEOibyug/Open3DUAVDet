import numpy as np
from uavdet3d.utils.object_encoder import all_object_encoders
from uavdet3d.utils.centernet_utils import draw_gaussian_to_heatmap, draw_res_to_heatmap
import torch
import cv2
from functools import partial
import copy
from scipy.spatial.transform import Rotation
import random

def create_photometric_augmentation(image):
    """
    使用numpy实现图像光度增强，复现Albumentations的功能

    Args:
        image: numpy数组，形状为(H, W, C)或(H, W)，像素值范围[0, 255]

    Returns:
        augmented_image: 增强后的图像
    """
    # 确保输入是numpy数组
    image = np.array(image, dtype=np.uint8)
    original_shape = image.shape

    # 如果是灰度图像，转换为三通道
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    result = image.copy()

    # 第一个OneOf组合 - CLAHE或RandomGamma (p=0.5)
    if np.random.random() < 0.5:
        if np.random.random() < 0.3:  # CLAHE (p=0.3)
            result = apply_clahe(result, clip_limit=2.0, tile_grid_size=(8, 8))
        elif np.random.random() < 0.7:  # RandomGamma (p=0.7)
            result = apply_random_gamma(result, gamma_limit=(80, 120))

    # 第二个OneOf组合 - 各种模糊 (p=0.4)
    if np.random.random() < 0.4:
        blur_choice = np.random.random()
        if blur_choice < 0.4:  # Blur (p=0.4)
            result = apply_blur(result, blur_limit=3)
        elif blur_choice < 0.7:  # MotionBlur (p=0.3)
            result = apply_motion_blur(result, blur_limit=3)
        else:  # GaussianBlur (p=0.3)
            result = apply_gaussian_blur(result, blur_limit=3)

    # 第三个OneOf组合 - 亮度对比度或色调饱和度 (p=0.8)
    if np.random.random() < 0.8:
        if np.random.random() < 0.7:  # RandomBrightnessContrast (p=0.7)
            result = apply_brightness_contrast(result,
                                               brightness_limit=0.2,
                                               contrast_limit=0.2)
        else:  # HueSaturationValue (p=0.3)
            result = apply_hsv(result,
                               hue_shift_limit=10,
                               sat_shift_limit=20,
                               val_shift_limit=20)

    # 高斯噪声 (p=0.2)
    if np.random.random() < 0.2:
        result = apply_gaussian_noise(result)

    # 如果原图是灰度图，转换回灰度
    if len(original_shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    return result


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """应用CLAHE（对比度限制自适应直方图均衡化）"""
    # 转换为LAB色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 只对L通道应用CLAHE
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # 转换回RGB
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result


def apply_random_gamma(image, gamma_limit=(80, 120)):
    """应用随机伽马校正"""
    gamma = np.random.uniform(gamma_limit[0] / 100.0, gamma_limit[1] / 100.0)

    # 归一化到[0,1]
    normalized = image.astype(np.float32) / 255.0

    # 应用伽马校正
    corrected = np.power(normalized, gamma)

    # 转换回[0,255]
    result = (corrected * 255).astype(np.uint8)
    return result


def apply_blur(image, blur_limit=3):
    """应用简单模糊"""
    kernel_size = np.random.randint(1, blur_limit + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    result = cv2.blur(image, (kernel_size, kernel_size))
    return result


def apply_motion_blur(image, blur_limit=3):
    """
    对图像应用运动模糊
    :param image: 输入图像 (numpy array, BGR)
    :param blur_limit: 最大模糊核大小（奇数，>=3）
    :return: 模糊后的图像
    """
    if blur_limit < 3:
        raise ValueError("blur_limit 必须 >= 3 且为奇数")

    # 随机选择模糊核大小
    ksize = random.randint(3, blur_limit)
    if ksize % 2 == 0:  # 保证奇数
        ksize += 1

    # 生成运动模糊卷积核
    kernel = np.zeros((ksize, ksize))
    direction = random.choice(["horizontal", "vertical", "diagonal"])

    if direction == "horizontal":
        kernel[ksize // 2, :] = np.ones(ksize)
    elif direction == "vertical":
        kernel[:, ksize // 2] = np.ones(ksize)
    else:  # diagonal
        np.fill_diagonal(kernel, 1)

    kernel = kernel / ksize

    # 应用滤波
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def apply_gaussian_blur(image, blur_limit=3):
    """应用高斯模糊"""
    kernel_size = np.random.randint(1, blur_limit + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    sigma = kernel_size / 3.0
    result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return result


def apply_brightness_contrast(image, brightness_limit=0.2, contrast_limit=0.2):
    """应用亮度和对比度调整"""
    # 随机亮度和对比度参数
    brightness = np.random.uniform(-brightness_limit, brightness_limit)
    contrast = np.random.uniform(1 - contrast_limit, 1 + contrast_limit)

    # 转换为float32进行计算
    img_float = image.astype(np.float32)

    # 应用对比度调整
    img_float = img_float * contrast

    # 应用亮度调整（按最大值调整）
    max_val = 255.0
    img_float = img_float + brightness * max_val

    # 限制在有效范围内
    img_float = np.clip(img_float, 0, 255)

    return img_float.astype(np.uint8)


def apply_hsv(image, hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20):
    """应用HSV色彩空间调整"""
    # 转换为HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # 随机调整参数
    hue_shift = np.random.uniform(-hue_shift_limit, hue_shift_limit)
    sat_shift = np.random.uniform(-sat_shift_limit, sat_shift_limit)
    val_shift = np.random.uniform(-val_shift_limit, val_shift_limit)

    # 应用调整
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180  # 色调是循环的
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)  # 饱和度
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val_shift, 0, 255)  # 明度

    # 转换回RGB
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return result


def apply_gaussian_noise(image, var_limit=(10.0, 50.0)):
    """应用高斯噪声"""
    # 限制噪声强度
    var_limit = (max(5.0, var_limit[0]), min(30.0, var_limit[1]))
    var = np.random.uniform(var_limit[0], var_limit[1])

    # 生成噪声
    noise = np.random.normal(0, var ** 0.5, image.shape)

    # 添加噪声
    noisy = image.astype(np.float32) + noise

    # 限制在有效范围内
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)



class DataAugmentor():
    def __init__(self,root_path, dataset_cfg, logger):

        self.root_path, self.dataset_cfg, self.logger = root_path, dataset_cfg,logger

        processor_configs = dataset_cfg

        self.data_processor_queue = []

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def resize_with_padding(self, img, boxes9d, this_boxes_2d, scale):
        H, W = img.shape[:2]

        new_W = int(W * scale)
        new_H = int(H * scale)

        resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros_like(img)

        start_x = (W - new_W) // 2
        start_y = (H - new_H) // 2
        end_x = start_x + new_W
        end_y = start_y + new_H

        sx, sy = max(start_x, 0), max(start_y, 0)
        ex, ey = min(end_x, W), min(end_y, H)
        resized_cropped = resized[max(-start_y, 0):max(-start_y, 0) + (ey - sy),
                          max(-start_x, 0):max(-start_x, 0) + (ex - sx)]

        canvas[sy:ey, sx:ex] = resized_cropped

        # 缩放 boxes9d
        boxes9d[:, 2] /= scale

        # 缩放 boxes_2d 并调整坐标以保持中心对齐
        if len(this_boxes_2d) > 0:
            # 首先缩放边界框坐标

            new_this_boxes_2d = this_boxes_2d.copy().astype(np.float32)
            new_this_boxes_2d *= scale  # 缩放坐标

            # 然后平移边界框以保持中心对齐
            offset_x = start_x
            offset_y = start_y
            new_this_boxes_2d[:, [0, 2]] += offset_x
            new_this_boxes_2d[:, [1, 3]] += offset_y

            # 确保边界框在图像范围内
            new_this_boxes_2d[:, [0, 2]] = np.clip(new_this_boxes_2d[:, [0, 2]], 0, W)
            new_this_boxes_2d[:, [1, 3]] = np.clip(new_this_boxes_2d[:, [1, 3]], 0, H)
        else:
            new_this_boxes_2d = this_boxes_2d

        return canvas, boxes9d, new_this_boxes_2d.astype(int)

    def flip_box(self,box):

        box = np.array(box)
        new_box = box.copy()
        new_box[0] = -new_box[0]
        new_box[7] = -new_box[7]
        new_box[8] = -new_box[8]
        return new_box

    def horizontal_fold_boxes(self, boxes, image_width):
        """
        对N个2D box进行水平对折

        Args:
            boxes: numpy array, shape (N, 4), 格式为 [x1, y1, x2, y2]
                   其中 (x1, y1) 是左上角，(x2, y2) 是右下角
            image_width: int, 图像宽度，用于确定对折轴位置

        Returns:
            folded_boxes: numpy array, shape (N, 4), 对折后的boxes
        """
        # 输入验证
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"boxes应该是shape为(N, 4)的数组，当前shape: {boxes.shape}")

        if len(boxes) == 0:
            return boxes.copy()

        if image_width <= 0:
            raise ValueError(f"image_width必须大于0，当前值: {image_width}")

        # 复制输入数组，避免修改原数组
        folded_boxes = boxes.copy().astype(float)

        # 对折轴位置（图像中心）
        fold_axis = image_width / 2.0

        # 提取坐标
        x1 = folded_boxes[:, 0]
        y1 = folded_boxes[:, 1]
        x2 = folded_boxes[:, 2]
        y2 = folded_boxes[:, 3]

        # 验证box格式的合理性
        invalid_boxes = (x1 >= x2) | (y1 >= y2)
        if np.any(invalid_boxes):
            print(f"警告: 发现 {np.sum(invalid_boxes)} 个无效的box (x1>=x2 或 y1>=y2)")
            # 可以选择修正或跳过无效的box
            valid_mask = ~invalid_boxes
            if not np.any(valid_mask):
                return np.array([]).reshape(0, 4)

        # 水平对折变换：x_new = 2 * fold_axis - x_old
        x1_folded = 2 * fold_axis - x1
        x2_folded = 2 * fold_axis - x2

        # 对折后需要重新确保 x1 < x2（因为对折后顺序会颠倒）
        folded_boxes[:, 0] = np.minimum(x1_folded, x2_folded)
        folded_boxes[:, 2] = np.maximum(x1_folded, x2_folded)

        # y坐标保持不变
        folded_boxes[:, 1] = y1
        folded_boxes[:, 3] = y2

        # 边界处理：将超出图像边界的坐标裁剪到有效范围
        folded_boxes[:, 0] = np.clip(folded_boxes[:, 0], 0, image_width)  # x1
        folded_boxes[:, 2] = np.clip(folded_boxes[:, 2], 0, image_width)  # x2

        # 检查对折后是否有box变得无效（宽度为0）
        zero_width_mask = folded_boxes[:, 0] >= folded_boxes[:, 2]
        if np.any(zero_width_mask):
            print(f"警告: 对折后有 {np.sum(zero_width_mask)} 个box宽度变为0或负数")

        return folded_boxes

    def merge_images_with_objects(self, image_with_object, image_background, boxes_2d, random_boxes_2d):
        """
        将带有目标的图像与背景图像合并，只保留目标区域

        参数:
        image_with_object: 包含目标的图像 (numpy数组)
        image_background: 背景图像 (numpy数组)
        boxes_2d: 目标边界框列表，格式为 [[x1, y1, x2, y2], ...]

        返回:
        合并后的图像
        """
        # 确保图像尺寸相同
        if image_with_object.shape[:2] != image_background.shape[:2]:
            image_background = cv2.resize(image_background, (image_with_object.shape[1], image_with_object.shape[0]))

        # 创建目标图像的掩码
        mask = np.zeros(image_with_object.shape[:2], dtype=np.uint8)

        # 为每个边界框创建掩码
        for box in boxes_2d:
            x1, y1, x2, y2 = box
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, image_with_object.shape[1]))
            y1 = max(0, min(y1, image_with_object.shape[0]))
            x2 = max(0, min(x2, image_with_object.shape[1]))
            y2 = max(0, min(y2, image_with_object.shape[0]))

            # 在掩码上绘制矩形区域（白色）
            cv2.rectangle(mask,(int(x1), int(y1)), (int(x2), int(y2)), 255, -1)


        mask2 = np.zeros(image_with_object.shape[:2], dtype=np.uint8)

        if len(boxes_2d)>0 and len(random_boxes_2d)==0:

            all_boxes =boxes_2d

        elif len(boxes_2d)==0 and len(random_boxes_2d)>0:
            all_boxes = random_boxes_2d

        elif len(boxes_2d) == 0 and len(random_boxes_2d)==0:
            return image_with_object
        else:
            all_boxes = np.concatenate([boxes_2d, random_boxes_2d])

        for box in all_boxes:
            x1, y1, x2, y2 = box
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, image_with_object.shape[1]))
            y1 = max(0, min(y1, image_with_object.shape[0]))
            x2 = max(0, min(x2, image_with_object.shape[1]))
            y2 = max(0, min(y2, image_with_object.shape[0]))

            # 在掩码上绘制矩形区域（白色）
            cv2.rectangle(mask2, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

        # 反转掩码以获取背景区域
        background_mask = cv2.bitwise_not(mask2)

        # 提取目标区域
        foreground = cv2.bitwise_and(image_with_object, image_with_object, mask=mask)

        # 提取背景区域
        background = cv2.bitwise_and(image_background, image_background, mask=background_mask)

        # 合并前景和背景
        result = cv2.add(foreground, background)

        return result


    def image_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_scaling, config=config)

        boxes9d = data_dict['boxes9d'] # 1,3,3
        this_boxes_2d = data_dict['this_boxes_2d']

        image = data_dict['image']

        scale_range = config.SCALE_RANGE

        this_scale = np.random.uniform(scale_range[0], scale_range[1])

        this_image = image

        new_image, new_boxes9d, this_boxes_2d = self.resize_with_padding(this_image, boxes9d, this_boxes_2d, this_scale)

        data_dict['image'] = new_image

        data_dict['boxes9d'] = new_boxes9d

        data_dict['this_boxes_2d'] = this_boxes_2d

        return data_dict

    def photometric_augmentation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.photometric_augmentation, config=config)

        image = data_dict['image']

        if np.random.randint(2):

            new_image = create_photometric_augmentation(image)

            data_dict['image'] = new_image.astype(np.float32)

        return data_dict

    def image_flipping(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_flipping, config=config)

        image = data_dict['image']
        boxes9d = data_dict['boxes9d']
        this_boxes_2d = data_dict['this_boxes_2d']

        H,W,_ = image.shape

        if np.random.randint(2): #
            new_boxes = []
            for box in boxes9d:
                new_boxes.append(self.flip_box(box))
            data_dict['boxes9d'] = np.array(new_boxes)
            data_dict['image'] = cv2.flip(image, 1)

            if 'this_boxes_2d' in data_dict:
                data_dict['this_boxes_2d'] = self.horizontal_fold_boxes(this_boxes_2d, image_width=W)
        else:
            return data_dict

        return data_dict

    def frame_mixing(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.frame_mixing, config=config)

        image = data_dict['image']
        this_boxes_2d = data_dict['this_boxes_2d']
        random_image = data_dict['random_image']
        random_boxes_2d = data_dict['random_boxes_2d']

        if np.random.randint(2):
            new_image = self.merge_images_with_objects(image, random_image, this_boxes_2d, random_boxes_2d)
            data_dict['image'] = new_image

        return data_dict

    def __call__(self, data_dict):

        for func in self.data_processor_queue:
            data_dict = func(data_dict)

        return data_dict