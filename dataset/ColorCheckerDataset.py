import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from typing import Union, List, Tuple
from .DataAugmenter import DataAugmenter as Augmenter
from .utils import normalize, bgr_to_rgb,linear_to_nonlinear, normalize_max
from config import TRAIN_IMG_H, TRAIN_IMG_W, TEST_IMG_H, TEST_IMG_W
import cv2
from torch.utils.data import DataLoader

"ColorChecker需要三折验证，folds_num=1-3分别跑一版，然后求平均"

class ColorCheckerDataset(data.Dataset):

    def __init__(self, mode: str = 'train', path: str = '/dataset/CC_full/', folds_num: int = 1):

        self.mode = mode
        self.da = Augmenter()
        self.base = path
        self.folds_path = os.path.join(self.base, "index", "folds.mat")
        self.metadata_path = os.path.join(self.base, "index", "metadata.txt")

        folds = scipy.io.loadmat(self.folds_path)
        metadata = open(self.metadata_path, 'r').readlines()

        self.train_files = [metadata[i - 1] for i in folds["tr_split"][0][folds_num][0]]
        self.test_files = [metadata[i - 1] for i in folds["te_split"][0][folds_num][0]]


    def __getitem__(self, index: int) -> Tuple:
        ""
        file = {'train': self.train_files, 'test': self.test_files}[self.mode][index]
        file_name = file.strip().split(' ')[1].split(".")[0]
        img = normalize_max(np.array(np.load(os.path.join(self.base, file_name + '.npy')), dtype='float32'))
        img = bgr_to_rgb(img)
        illu = np.array(np.load(os.path.join(self.base, file_name + '_gt.npy')), dtype='float32')
        mask = np.array(np.load(os.path.join(self.base, file_name + '_mask.npy')), dtype='float32')

        img = img * mask
        img_gt = self._correct_full(img, illu)

        if self.mode == 'train':
            img, img_gt, illu = self.da.augment(img, img_gt, illu)
        else:
            img = cv2.resize(img,[TEST_IMG_H, TEST_IMG_W])
            img_gt = cv2.resize(img_gt, [TEST_IMG_H, TEST_IMG_W])

        # plt.imshow(linear_to_nonlinear(img))
        # plt.show()

        return self._to_tensor(img), self._to_tensor(img_gt), torch.tensor(illu), file_name


    def __len__(self) -> int:
        file_len = {'train': len(self.train_files),
                    'test': len(self.test_files)}[self.mode]
        return file_len


    def _correct_full(self, img: np.ndarray, illum: np.ndarray) -> np.ndarray:
        """对整图做白平衡校正"""
        eps = 1e-10
        out = np.zeros_like(img)
        for c in range(3):
            out[..., c] = img[..., c] / (illum[c] + eps)
        out_ = out/np.max(out)
        return out_

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """从 HWC 转 CHW 并 gamma 校正到 [0,1] 再转 Tensor"""
        arr = np.power(arr, 1 / 2.2)
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr.copy())





