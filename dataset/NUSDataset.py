import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from typing import Union, List, Tuple
from .DataAugmenter import DataAugmenter as Augmenter
from .utils import normalize, bgr_to_rgb,linear_to_nonlinear, normalize_max
import random
import glob
import cv2
from config import TRAIN_IMG_H, TRAIN_IMG_W, TEST_IMG_H, TEST_IMG_W

from torch.utils.data import DataLoader

"camera list: Canon1DsMkIII, Canon600D, FujifilmXM1, NikonD5200, OlympusEPL6, PanasonicGX1, SamsungNX2000, SonyA57, NikonD40(无)"

class NUSDataset(data.Dataset):

    def __init__(self, mode: str, path: str, folds: Union[int,str], camera_name: Union[str,bool], supported_cameras: Union[str,bool]):
        ""
        "camera_name:测试的相机; supported_cameras: 单一相机"
        "folds: int + supported_cameras: 单一相机内三折"
        "folds: str(all): 全部训练"
        "folds: str(exclude) + camera_name: 除camera_name的训练，camera_name测试"

        self.mode = mode
        self.da = Augmenter()

        self.camera_name = camera_name
        self.supported_cameras = supported_cameras
        self.folds = folds

        self.base = path
        self.fold_path = os.path.join(self.base,'data_fold')
        if isinstance(self.folds, str):
            "all/exclude"
            self.train_files, self.test_files = self._merge_multi_camera_files(self.fold_path, self.camera_name)
        elif isinstance(self.supported_cameras, str):
            "0,1,2"
            self.train_files, self.test_files = self._merge_single_camera_files(self.fold_path, self.supported_cameras,self.folds)


    def __getitem__(self, index: int) -> Tuple:
        ""
        file_name = {'train': self.train_files, 'test': self.test_files}[self.mode][index]

        img = np.array(np.load(os.path.join(self.base, file_name + '.npy')), dtype='float32')
        img = bgr_to_rgb(img)
        img = img * (1/np.max(img))              ## 归一化


        illu = np.array(np.load(os.path.join(self.base, file_name + '_gt.npy')), dtype='float32')
        mask = np.array(np.load(os.path.join(self.base, file_name + '_mask.npy')), dtype='float32')

        img = img * mask
        img_gt = self._correct_full(img, illu)

        if self.mode == 'train':
            img, img_gt, illu = self.da.augment(img, img_gt, illu)
        else:
            img = cv2.resize(img,[TEST_IMG_H, TEST_IMG_W])
            img_gt = cv2.resize(img_gt, [TEST_IMG_H, TEST_IMG_W])

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


    def _merge_multi_camera_files(self, data_fold_path, camera_name):
        """
        读取data_fold中所有相机文件夹的txt文件并合并
        """
        train_files, test_files = [],[]
        all_camera_folders = ['Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 'NikonD5200', 'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57']

        "去除测试的相机fold"
        if isinstance(camera_name, str):
            all_camera_folders = [cam for cam in all_camera_folders if cam != camera_name]

        # 读取训练数据
        for camera in all_camera_folders:
            camera_path = os.path.join(data_fold_path, camera)
            if os.path.isdir(camera_path):
                train_files.extend(self._read_txt_files(camera_path))

        # 读取测试数据
        test_camera_path = os.path.join(data_fold_path, camera_name)
        if os.path.isdir(test_camera_path):
            test_files.extend(self._read_txt_files(test_camera_path))

        return np.array(train_files), np.array(test_files)



    def _merge_single_camera_files(self, data_fold_path, camera_name, test_fold):
        """
        单相机文件合并：使用fold交叉验证，test_fold作为测试集，其余作为训练集
        """
        train_files = []
        test_files = []
        all_folds = ['1', '2', '3']

        camera_path = os.path.join(data_fold_path, camera_name)

        # 读取训练数据（除test_fold外的所有fold）
        train_folds = [fold for fold in all_folds if fold != str(test_fold)]
        for fold in train_folds:
            fold_file = os.path.join(camera_path, f"fold{fold}.txt")
            train_files.extend(self._read_single_file(fold_file))

        # 读取测试数据
        test_file = os.path.join(camera_path, f"fold{test_fold}.txt")
        test_files.extend(self._read_single_file(test_file))

        return np.array(train_files), np.array(test_files)


    def _read_txt_files(self, folder_path):
        """读取文件夹中所有txt文件的内容"""
        files = []
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # 移除文件扩展名
                            base_name = os.path.splitext(line)[0]
                            files.append(base_name)
            except Exception as e:
                print(f"读取文件 {txt_file} 时出错: {e}")

        return files

    def _read_single_file(self, file_path):
        """读取单个txt文件的内容"""
        files = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        base_name = os.path.splitext(line)[0]
                        files.append(base_name)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")

        return files






