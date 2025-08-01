import math
import random

import cv2
import numpy as np

from config import TRAIN_IMG_H, TRAIN_IMG_W, TEST_IMG_H, TEST_IMG_W



class DataAugmenter:

    def __init__(self):
        # Input Size of the fully-convolutional network (SqueezeNet)
        self.__train_size = (TRAIN_IMG_H, TRAIN_IMG_W)
        self.__test_size = (TEST_IMG_H, TEST_IMG_W)

        # Rotation angle
        self.__angle = 60

        # Patch scale
        self.__scale = [0.1, 1.0]

        # Color rescaling
        self.__color = 0.8



    @staticmethod
    def __rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle (in degrees).
        The returned image will be large enough to hold the entire new image, with a black background
        """

        # Get the image size (note: NumPy stores image matrices backwards)
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

        rot_mat_no_translate = np.matrix(rot_mat[0:2, 0:2])

        image_w2, image_h2 = image_size[0] * 0.5, image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_no_translate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_no_translate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_no_translate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_no_translate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos, x_neg = [x for x in x_coords if x > 0], [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos, y_neg = [y for y in y_coords if y > 0], [y for y in y_coords if y < 0]

        right_bound, left_bound, top_bound, bot_bound = max(x_pos), min(x_neg), max(y_pos), min(y_neg)
        new_w, new_h = int(abs(right_bound - left_bound)), int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)], [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

        # Compute the transform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        return cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

    @staticmethod
    def __largest_rotated_rect(w: float, h: float, angle: float) -> tuple:
        """
        Given a rectangle of size w x h that has been rotated by 'angle' (in radians), computes the width and height of
        the largest possible axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow. Converted to Python by Aaron Snoswell
        """
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        length = h if (w < h) else w
        d = length * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
        delta = math.pi - alpha - gamma
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return bb_w - 2 * x, bb_h - 2 * y

    def __crop_around_center(self, image: np.ndarray, width: float, height: float) -> np.ndarray:
        """ Given a NumPy / OpenCV 2 image, crops it to the given width and height around it's centre point """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        width = image_size[0] if width > image_size[0] else width
        height = image_size[1] if height > image_size[1] else height

        x1, x2 = int(image_center[0] - width * 0.5), int(image_center[0] + width * 0.5)
        y1, y2 = int(image_center[1] - height * 0.5), int(image_center[1] + height * 0.5)

        return cv2.resize(image[y1:y2, x1:x2], self.__train_size)

    def __rotate_and_crop(self, image: np.ndarray, angle: float) -> np.ndarray:
        width, height = image.shape[:2]
        target_width, target_height = self.__largest_rotated_rect(width, height, math.radians(angle))
        return self.__crop_around_center(self.__rotate_image(image, angle), target_width, target_height)

    @staticmethod
    def __random_flip(img: np.ndarray) -> np.ndarray:
        """ Perform random left/right flip with probability 0.5 """
        if random.randint(0, 1):
            img = img[:, ::-1]
        return img.astype(np.float32)

    def augment(self, img: np.ndarray,  img_gt: np.ndarray, illumination: np.ndarray) -> tuple:
        """
        Args:
            img:      原始输入图像 (H×W×3)
            illumination: 光照向量或矩阵
            img_gt:   对应的 ground-truth 图像 (H×W×3)
            cube:     是否在“cube”模式下处理 illuminant
        Returns:
            new_img:     增强后的 img
            new_illum:   增强后的 illumination
            new_img_gt:  增强后的 img_gt（与 new_img 做相同变换）
        """
        # 1. 随机比例 crop
        scale = math.exp(random.random() * math.log(self.__scale[1] / self.__scale[0])) * self.__scale[0]
        s = min(max(int(round(min(img.shape[:2]) * scale)), 10), min(img.shape[:2]))
        start_x = random.randrange(0, img.shape[0] - s + 1)
        start_y = random.randrange(0, img.shape[1] - s + 1)

        img     = img   [start_x:start_x + s, start_y:start_y + s]
        img_gt  = img_gt[start_x:start_x + s, start_y:start_y + s]

        # 2. 随机旋转并 crop 到 train_size
        angle = (random.random() - 0.5) * self.__angle
        img     = self.__rotate_and_crop(img,    angle)
        img_gt  = self.__rotate_and_crop(img_gt, angle)



        # 3. 随机左右翻转
        if random.randint(0, 1):
            img    = img   [:, ::-1]
            img_gt = img_gt[:, ::-1]

        img     = img.astype(np.float32)
        img_gt  = img_gt.astype(np.float32)

        # 4. 颜色强度缩放（同样参数同时作用于 img 和 img_gt）
        #    这里只做通道缩放，不打乱通道顺序
        scale_r = 1 + (random.random() * self.__color - 0.5 * self.__color)
        scale_g = 1 + (random.random() * self.__color - 0.5 * self.__color)
        scale_b = 1 + (random.random() * self.__color - 0.5 * self.__color)

        img    *= np.array([scale_r, scale_g, scale_b], dtype=np.float32).reshape(1,1,3)
        # img_gt *= np.array([scale_r, scale_g, scale_b], dtype=np.float32).reshape(1,1,3)

        new_img    = np.clip(img,    0, 65535)


        new_img_gt = np.clip(img_gt, 0, 65535)

        # 5. 处理 illuminant

        new_illuminant = np.zeros_like(illumination)
        color_aug = np.diag([scale_r, scale_g, scale_b]).astype(np.float32)
        for i in range(3):
            for j in range(3):
                new_illuminant[i] += illumination[j] * color_aug[i, j]

        new_illumination = np.clip(new_illuminant, 0.01, 100)
        return new_img, new_img_gt, new_illumination

    def crop(self, img: np.ndarray, scale: float = 0.5) -> np.ndarray:
        return cv2.resize(img, self.__test_size, fx=scale, fy=scale)

    def rgb_to_bgr(self, x: np.ndarray) -> np.ndarray:
        return x[::-1]
