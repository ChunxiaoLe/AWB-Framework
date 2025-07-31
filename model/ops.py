import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
import torch
from torch import Tensor
import math

def imshow_tensor(
        tensor: Union[np.ndarray, "torch.Tensor", "tf.Tensor"],
        mean: Optional[Tuple[float, float, float]] = (0.485, 0.456, 0.406),
        std: Optional[Tuple[float, float, float]] = (0.229, 0.224, 0.225),
        denormalize: bool = False,
        figsize: Tuple[int, int] = (6, 6),
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    显示张量格式的图像（支持PyTorch/TensorFlow/NumPy）

    参数:
        tensor: 输入图像张量，支持形状:
                (C, H, W) - PyTorch格式
                (H, W, C) - TensorFlow格式
                (H, W)    - 灰度图像
        mean: 归一化时使用的均值（每个通道）
        std: 归一化时使用的标准差（每个通道）
        denormalize: 是否执行反归一化操作
        figsize: 图像显示尺寸（当ax=None时有效）
        title: 图像标题
        ax: matplotlib轴对象，如果为None则创建新图

    返回:
        matplotlib轴对象
    """
    # 检查张量类型并转换为NumPy数组
    if "torch" in str(type(tensor)):
        tensor = tensor.detach().cpu().numpy()
    elif "tensorflow" in str(type(tensor)) or "tensor" in str(type(tensor)).lower():
        tensor = tensor.numpy()

    # 处理不同形状的输入
    if tensor.ndim == 4:  # 批次维度 (B, C, H, W) 或 (B, H, W, C)
        tensor = tensor[0]  # 取批次中的第一张图像

    # 通道处理
    if tensor.ndim == 3:
        # PyTorch格式 (C, H, W) -> (H, W, C)
        if tensor.shape[0] in [1, 3]:  # 通道在前
            tensor = tensor.transpose(1, 2, 0)
    elif tensor.ndim == 2:  # 灰度图
        tensor = np.expand_dims(tensor, axis=-1)

    # 反归一化处理
    if denormalize and mean is not None and std is not None:
        # 确保mean/std转换为NumPy数组
        mean = np.array(mean).reshape(1, 1, -1)
        std = np.array(std).reshape(1, 1, -1)

        # 仅当通道数匹配时执行反归一化
        if tensor.shape[-1] == mean.shape[-1]:
            tensor = tensor * std + mean

    # 数值裁剪到[0, 1]范围
    tensor = np.clip(tensor, 0, 1)

    # 创建图像显示
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # 显示图像
    if tensor.shape[-1] == 1:  # 单通道灰度图
        ax.imshow(tensor[:, :, 0], cmap='gray', vmin=0, vmax=1)
    else:  # 多通道彩色图
        ax.imshow(tensor)

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    plt.show()


def awb(img: Tensor, illuminant: Tensor) -> Tensor:
    # Ensure data is on the same device


    # Compute per-channel correction factor: [B, 3, 1, 1]
    # Multiply by sqrt(3) to maintain overall brightness
    correction = illuminant.view(-1, 3, 1, 1) * math.sqrt(3)

    # Apply gain correction
    corrected_img = img / (correction + 1e-10)

    # Normalize each sample by its maximum value across all channels and spatial dims
    # resulting in values in [0, 1]
    max_val = corrected_img.amax(dim=[1, 2, 3], keepdim=True)
    max_val = max_val + 1e-10
    normalized_img = corrected_img / max_val

    return normalized_img  # Tensor of shape [B, 3, H, W]


def batch_masked_rgb_mean(
        images: torch.Tensor,
        masks: torch.Tensor,
        epsilon: float = 1e-8
) -> torch.Tensor:
    """
    计算批次图像中掩码区域(mask>0.5)的RGB平均值 (GPU优化版本)

    参数:
        images: 输入图像张量，形状为 (B, C, H, W) 或 (B, H, W, C)
        masks: 掩码张量，形状为 (B, 1, H, W) 或 (B, H, W) 或 (B, H, W, 1)
        epsilon: 防止除以零的小常数

    返回:
        形状为 (B, 3) 的张量，包含每个图像的RGB平均值

    处理流程:
        1. 统一图像格式为 (B, C, H, W)
        2. 统一掩码格式为 (B, H, W)
        3. 计算每个图像的掩码区域RGB平均值
        4. 处理掩码区域为空的情况
    """
    # 确保输入在相同设备上
    masks = masks.to(images.device)

    # 统一图像格式为 (B, C, H, W)
    if images.dim() == 4 and images.shape[3] in [1, 3]:  # (B, H, W, C) 格式
        images = images.permute(0, 3, 1, 2)

    # 统一掩码格式为 (B, H, W)
    if masks.dim() == 4:  # (B, 1, H, W) 或 (B, H, W, 1)
        if masks.shape[1] == 1:  # (B, 1, H, W)
            masks = masks.squeeze(1)
        elif masks.shape[3] == 1:  # (B, H, W, 1)
            masks = masks.squeeze(3)

    # 验证形状
    B, C, H, W = images.shape
    if masks.shape != (B, H, W):
        raise ValueError(f"掩码形状 {masks.shape} 与图像形状 {(B, H, W)} 不匹配")

    # 创建二进制掩码 (0 或 1)
    binary_masks = (masks > 0.5).float()

    # 计算每个图像的掩码像素数量
    mask_pixel_counts = binary_masks.sum(dim=(1, 2))  # (B,)

    # 处理完全空的掩码（避免除零）
    empty_masks = (mask_pixel_counts < epsilon)
    mask_pixel_counts = mask_pixel_counts.clamp(min=epsilon)

    # 计算每个通道的掩码区域总和
    channel_sums = torch.zeros(B, C, device=images.device)
    for c in range(C):
        channel_sums[:, c] = (images[:, c] * binary_masks).sum(dim=(1, 2))

    # 计算通道平均值
    channel_means = channel_sums / mask_pixel_counts.unsqueeze(1)

    # 处理灰度图像：复制单通道值为RGB
    if C == 1:
        rgb_means = channel_means.repeat(1, 3)  # (B, 3)
    # 处理RGB图像：取前三个通道
    elif C >= 3:
        rgb_means = channel_means[:, :3]  # (B, 3)
    else:
        raise ValueError(f"不支持的通道数: {C}")

    # 将完全空的掩码结果设为0
    rgb_means[empty_masks] = 0

    return rgb_means  # (B, 3)