import os
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import re

from .DataAugmenter import DataAugmenter as Augmenter
from config import TRAIN_IMG_H, TRAIN_IMG_W, TEST_IMG_H, TEST_IMG_W
from torch import Tensor
from PIL import Image

class CubeDataset(Dataset):
    def __init__(self, mode: str = 'train', path: str='/dataset/cube_raw'):
        self.mode = mode

        self.da = Augmenter()

        base = path
        self.data_dir     = base
        self.mask_dir     = os.path.join(base, 'mask')
        self.vic_path     = os.path.join(base, 'indicator', 'VIC.png')
        self.folds_path   = os.path.join(base, 'index', 'img.txt')
        self.gt_path      = os.path.join(base, 'index', 'cube+_gt_new.txt')
        self.mask_index   = os.path.join(base, 'index', 'updated_cube_mask.txt')

        # 划分 train/test，及测试集中的纯色场景
        self.train_files, self.test_files = self._read_splits(self.folds_path)

        # 读取 GT 数据
        self.train_gt, self.test_gt = self._read_gt(self.gt_path, self.train_files, self.test_files)

    def __len__(self) -> int:
        file_len = {'train': len(self.train_files),
                    'test': len(self.test_files)}[self.mode]
        return file_len


    def __getitem__(self, idx: int):
        files = {'train': self.train_files, 'test': self.test_files}[self.mode]
        gts = {'train': self.train_gt, 'test': self.test_gt}[self.mode]
        fname = files[idx]
        illu  = gts[idx].astype('float32')

        img = self._load_image(os.path.join(self.data_dir, fname))

        "遮盖住立方蜘蛛"
        img_gt = self._correct_full(img, illu)
        if self.mode == 'train':
            img, img_gt, illu = self.da.augment(img, img_gt, illu)

        else:
            "resize"
            img_gt = cv2.resize(img_gt, [TEST_IMG_H, TEST_IMG_W])
            img = cv2.resize(img, [TEST_IMG_H, TEST_IMG_W])

        # plt.imshow(img)
        # plt.title(fname+'input')
        # plt.show()

        return self._to_tensor(img), self._to_tensor(img_gt), torch.from_numpy(illu), fname


    def _load_image(self, path: str, black: int = 2048, white: int = 14582) -> np.ndarray:
        img = imageio.imread(path, 'PNG-FI').astype(np.float32)
        img = np.maximum(img - black, 0)
        # img = img / (white - black)
        img = np.clip(img, 0.0, white-black) * (1.0 / np.max(img))
        return img

    def _correct_full(self, img: np.ndarray, illum: np.ndarray) -> np.ndarray:
        """对整图做白平衡校正"""
        eps = 1e-10
        out = np.zeros_like(img)
        for c in range(3):
            out[..., c] = img[..., c] / (illum[c] + eps)
        return np.clip(out, 0, 1)

    def _mask_colorchecker_corner(self, img: np.ndarray, size: int = 700) -> np.ndarray:
        h, w, _ = img.shape
        img[-size:, -size:, :] = 0
        return img

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """从 HWC 转 CHW 并 gamma 校正到 [0,1] 再转 Tensor"""
        arr = np.power(arr, 1/2.2)
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr.copy())

    def _read_splits(self, split_txt: str, ) -> Tuple[np.ndarray, np.ndarray]:
        """读取 train/test 划分；如果提供 pure_txt，则额外返回测试集中的纯色文件名"""
        split_line =  '.'
        with open(split_txt, 'r') as f:
            files = [l.strip() for l in f if l.strip()]
        train = [f for f in files if int(f.split(split_line)[0]) <= 1000]
        test = [f for f in files if int(f.split(split_line)[0]) > 1000]

        return np.array(train), np.array(test)



    def _read_gt(self, gt_filepath: str, train_list: np.ndarray,
                 test_list: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:


        # --- 3. 读取 GT 全部行 ---
        with open(gt_filepath, 'r') as f:
            gt_raw_lines = [
                list(map(float, line.strip().split()))
                for line in f
                if line.strip()  # 过滤空行
            ]

        # --- 4. 按图像顺序遍历，分配到训练/测试/纯色测试 ---
        train_vals = []
        test_vals = []
        pure_test_vals = []

        dele_train_list, dele_test_list, dele_pure_list = [], [], []

        for train_name in train_list:
            line = int(train_name.split('.')[0])-1
            if line < len(gt_raw_lines):
                train_vals.append(gt_raw_lines[line])
            else:
                dele_train_list.append(train_name)

        for test_name in test_list:
            line = int(test_name.split('.')[0])-1
            if line < len(gt_raw_lines):
                test_vals.append(gt_raw_lines[line])
            else:
                dele_test_list.append(test_name)


        return np.array(train_vals, dtype=float), np.array(test_vals, dtype=float)

    def _linear_to_nonlinear(self,img: [np.array, Image, Tensor]):
        if isinstance(img, np.ndarray):
            return np.power(img, (1.0 / 2.2))
        if isinstance(img, Tensor):
            return torch.pow(img, 1.0 / 2.2)

        # return F.to_pil_image(torch.pow(F.to_tensor(img), 1.0 / 2.2).squeeze(), mode="RGB")
