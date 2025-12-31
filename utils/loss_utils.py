#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from utils.graphics_utils import patch_offsets, patch_warp
from utils.sh_utils import eval_sh_point

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask


def compute_epipolar_consistency_loss(
        gt_image_gray, nearest_cam, viewpoint_cam, pixels, weights, d_mask, sample_num, patch_size,
        total_patch_size, ncc_weight
):
    """
    通过 Epipolar 几何约束计算光度一致性损失。
    """
    with torch.no_grad():
        # gt_image_gray: [H, W] 灰度图像 (目标图像)
        # nearest_cam, viewpoint_cam: 相机对象，包含变换矩阵和内参信息
        # pixels: [N, 2] 原始图像中的像素点坐标
        # weights: [N] 每个像素点的权重
        # d_mask: [N] 掩码数组，表示是否有效
        # sample_num: int 最大采样数量
        # patch_size: int 采样块的大小
        # total_patch_size: int patch_size * patch_size
        # ncc_weight: float LNCC 损失权重

        # Sample mask
        d_mask = d_mask.reshape(-1)  # [N]
        valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]  # [M], M = d_mask.sum()
        if d_mask.sum() > sample_num:
            index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace=False)  # [sample_num]
            valid_indices = valid_indices[index]  # [sample_num]

        weights = weights.reshape(-1)[valid_indices]  # [sample_num]
        pixels = pixels.reshape(-1, 2)[valid_indices]  # [sample_num, 2]

        offsets = patch_offsets(patch_size, pixels.device)  # [patch_size*patch_size, 2]
        ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()  # [sample_num, patch_size*patch_size, 2]

        H, W = gt_image_gray.squeeze().shape  # H, W: 图像高度和宽度
        pixels_patch = ori_pixels_patch.clone()  # [sample_num, patch_size*patch_size, 2]
        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0

        ref_gray_val = F.grid_sample(
            gt_image_gray.unsqueeze(1),  # [1, 1, H, W]
            pixels_patch.view(1, -1, 1, 2),  # [1, sample_num*patch_size*patch_size, 1, 2]
            align_corners=True
        )  # [1, 1, sample_num*patch_size*patch_size, 1]
        ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)  # [sample_num, patch_size*patch_size]

        # Compute fundamental matrix
        F_matrix = compute_fundamental_matrix(
            nearest_cam.world_view_transform, viewpoint_cam.world_view_transform,  # [4, 4]
            nearest_cam.get_k(nearest_cam.ncc_scale), viewpoint_cam.get_k(viewpoint_cam.ncc_scale)  # [3, 3]
        )  # [3, 3]

        # Compute epipolar lines in the nearest view
        pixels_homo = torch.cat([pixels, torch.ones_like(pixels[:, :1])], dim=-1)  # [sample_num, 3]
        epipolar_lines = torch.matmul(F_matrix, pixels_homo.T).T  # [sample_num, 3]

        # Compute nearest frame patch based on epipolar line sampling
        nearest_pixels_patch = project_onto_epipolar_lines(pixels, epipolar_lines, W, H, patch_size)  # [sample_num, patch_size*patch_size, 2]
        nearest_pixels_patch[:, :, 0] = 2 * nearest_pixels_patch[:, :, 0] / (W - 1) - 1.0
        nearest_pixels_patch[:, :, 1] = 2 * nearest_pixels_patch[:, :, 1] / (H - 1) - 1.0

        _, nearest_image_gray = nearest_cam.get_image()  # nearest_image_gray: [H, W]
        sampled_gray_val = F.grid_sample(
            nearest_image_gray[None],  # [1, 1, H, W]
            nearest_pixels_patch.reshape(1, -1, 1, 2),  # [1, sample_num*patch_size*patch_size, 1, 2]
            align_corners=True
        )  # [1, 1, sample_num*patch_size*patch_size, 1]
        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)  # [sample_num, patch_size*patch_size]

        # Compute LNCC loss
        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)  # ncc: [sample_num], ncc_mask: [sample_num]
        mask = ncc_mask.reshape(-1)  # [sample_num]
        ncc = ncc.reshape(-1) * weights  # [sample_num]
        ncc = ncc[mask].squeeze()  # [valid_sample_num]

        # if mask.sum() > 0:
        #     ncc_loss = ncc_weight * ncc.mean()  # Scalar
        #     # loss += ncc_loss
        ncc_loss = ncc_weight * ncc.mean()  # Scalar

    return ncc_loss



def compute_fundamental_matrix(R1, R2, K1, K2):
    """
    使用相机外参和内参计算基础矩阵。
    """

    def skew_symmetric(T):
        """
        将向量 T 转换为反对称矩阵 [T]_x
        """
        return torch.tensor([
            [0, -T[2], T[1]],
            [T[2], 0, -T[0]],
            [-T[1], T[0], 0]
        ], device=T.device, dtype=T.dtype)

    T = R2[3, :3] - R1[3, :3]
    R = R2[:3, :3] @ R1[:3, :3].T
    E = torch.cross(skew_symmetric(T), R)  # 计算本质矩阵
    F = torch.inverse(K2).T @ E @ torch.inverse(K1)
    return F


def project_onto_epipolar_lines(points, lines, width, height, patch_size):
    """
    根据极线方程，将像素投影到极线附近并生成用于计算一致性损失的像素块坐标。

    参数:
        points: torch.Tensor, [N, 2] 输入图像中的像素坐标 (x1, y1)。
        lines: torch.Tensor, [N, 3] 极线方程，形式为 ax + by + c = 0。
        width: int, 目标图像的宽度。
        height: int, 目标图像的高度。
        patch_size: int, 需要采样的像素块大小。

    返回:
        sampled_points: torch.Tensor, [N, patch_size*patch_size, 2] 极线附近采样的像素块坐标。
    """
    # num_points = points.shape[0]
    # half_patch = patch_size // 2

    # 创建采样的相对偏移
    offsets = patch_offsets(patch_size, points.device).squeeze(0)  # [patch_size*patch_size, 2]

    # 当前极线的参数 ax + by + c = 0
    a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]  # [N]

    # 点 (x1, y1) 投影到极线上的最近点 (x', y')
    x1, y1 = points[:, 0], points[:, 1]  # [N]
    denom = a ** 2 + b ** 2 + 1e-8  # 避免除零
    x_proj = (b * (b * x1 - a * y1) - a * c) / denom  # [N]
    y_proj = (a * (-b * x1 + a * y1) - b * c) / denom  # [N]

    # 将 x_proj 和 y_proj 扩展以匹配 offsets 的形状
    x_proj = x_proj.unsqueeze(1)  # [N, 1]
    y_proj = y_proj.unsqueeze(1)  # [N, 1]

    # 将偏移量广播到所有点
    sampled_points = torch.stack(
        [
            x_proj + offsets[:, 0],  # x 坐标
            y_proj + offsets[:, 1],  # y 坐标
        ],
        dim=-1,  # [N, patch_size*patch_size, 2]
    )

    # 保证采样点不超出图像边界
    sampled_points[..., 0] = torch.clamp(sampled_points[..., 0], 0, width - 1)  # x 坐标限制
    sampled_points[..., 1] = torch.clamp(sampled_points[..., 1], 0, height - 1)  # y 坐标限制

    return sampled_points



def compute_Homography_consistency_loss(
        gt_image_gray, nearest_cam, viewpoint_cam, pixels, weights, render_pkg, d_mask, sample_num, patch_size,
        total_patch_size, ncc_weight
):
    """
    通过 Homography 几何约束计算光度一致性损失。
    """
    with torch.no_grad():
        ## sample mask
        d_mask = d_mask.reshape(-1)  # 将深度掩码拉平成一维张量
        valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[
            d_mask]  # 筛选深度掩码中有效像素的索引
        if d_mask.sum() > sample_num:  # 如果有效像素数大于采样数
            index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num,
                                     replace=False)  # 随机选择指定数量的有效像素
            valid_indices = valid_indices[index]  # 更新为随机选择的有效像素索引

        weights = weights.reshape(-1)[valid_indices]  # 根据有效索引筛选权重
        ## sample ref frame patch
        pixels = pixels.reshape(-1, 2)[valid_indices]  # 根据有效索引筛选像素坐标
        offsets = patch_offsets(patch_size, pixels.device)  # 生成用于构建图像块的偏移
        ori_pixels_patch = pixels.reshape(-1, 1,
                                          2) / viewpoint_cam.ncc_scale + offsets.float()  # 计算原始图像块像素坐标

        H, W = gt_image_gray.squeeze().shape  # 获取参考图像的高宽
        pixels_patch = ori_pixels_patch.clone()  # 克隆原始像素块坐标
        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0  # 将像素横坐标归一化到[-1, 1]
        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0  # 将像素纵坐标归一化到[-1, 1]
        ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2),
                                     align_corners=True)  # 从参考灰度图采样图像块的灰度值
        ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)  # 调整采样灰度值的形状

        ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1,
                                                                               -2) @ viewpoint_cam.world_view_transform[
                                                                                     :3,
                                                                                     :3]  # 计算从参考视点到最近视点的旋转矩阵
        ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,
                                                 :3] + nearest_cam.world_view_transform[3,
                                                       :3]  # 计算从参考视点到最近视点的平移向量

        ## compute Homography
        ref_local_n = render_pkg["rendered_normal"].permute(1, 2, 0)  # 提取渲染的法向量图，并调整维度顺序
        ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]  # 筛选有效像素的法向量

        ref_local_d = render_pkg['rendered_distance'].squeeze()  # 提取渲染的深度图
        ref_local_d = ref_local_d.reshape(-1)[valid_indices]  # 筛选有效像素的深度值
        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(
                                ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
                                ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(
                                    0, 2, 1)) / ref_local_d[..., None, None]  # 计算参考视点到最近视点的单应矩阵
        H_ref_to_neareast = torch.matmul(
            nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3),
            H_ref_to_neareast)  # 转换到最近视点的像素坐标系
        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(
            viewpoint_cam.ncc_scale)  # 转换到参考视点的像素坐标系

        ## compute neareast frame patch
        grid = patch_warp(H_ref_to_neareast.reshape(-1, 3, 3),
                          ori_pixels_patch)  # 利用单应矩阵计算最近视点图像块的像素坐标
        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0  # 将最近视点像素横坐标归一化到[-1, 1]
        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0  # 将最近视点像素纵坐标归一化到[-1, 1]
        _, nearest_image_gray = nearest_cam.get_image()  # 获取最近视点的灰度图像
        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2),
                                         align_corners=True)  # 从最近视点灰度图采样图像块的灰度值
        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)  # 调整采样灰度值的形状

        ## compute loss
        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)  # 计算图像块之间的归一化互相关系数（LNCC）和有效掩码
        mask = ncc_mask.reshape(-1)  # 拉平成一维有效掩码
        ncc = ncc.reshape(-1) * weights  # 加权处理归一化互相关系数
        ncc = ncc[mask].squeeze()  # 仅保留有效像素的归一化互相关系数

        if mask.sum() > 0:  # 如果存在有效像素
            ncc_loss = ncc_weight * ncc.mean()  # 计算归一化互相关损失
            # loss += ncc_loss  # 累加到总损失
        else:
            ncc_loss = 0
        # ncc_loss = ncc_weight * ncc.mean()

    return ncc_loss

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img


def haar_wavelet_2d(image_tensor):
    """
    使用 Haar 小波变换分解 2D 图像，与 PyWavelets 的输出一致。
    支持奇数形状的输入图像。

    Args:
        image_tensor (torch.Tensor): 输入图像张量，形状为 [B, C, H, W]

    Returns:
        tuple: (低频分量 LL, (水平高频分量 LH, 垂直高频分量 HL, 对角高频分量 HH)),
               以及填充信息 (pad_h, pad_w)
    """
    B, C, H, W = image_tensor.shape
    # weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(image_tensor.device)
    # gray_image = torch.sum(image_tensor * weights, dim=1, keepdim=True)  # 转为灰度图 [B, 1, H, W]

    # 计算需要填充的大小
    pad_h = H % 2
    pad_w = W % 2

    # 如果 H 或 W 为奇数，进行对称填充
    if pad_h or pad_w:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    # 转为灰度图
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(image_tensor.device)
    gray_image = torch.sum(image_tensor * weights, dim=1, keepdim=True)

    # Haar 小波滤波器
    low_pass = torch.tensor([1 / torch.sqrt(torch.tensor(2.0)), 1 / torch.sqrt(torch.tensor(2.0))]).to(image_tensor.device)
    high_pass = torch.tensor([1 / torch.sqrt(torch.tensor(2.0)), -1 / torch.sqrt(torch.tensor(2.0))]).to(image_tensor.device)

    # 1D 滤波器扩展为 2D
    low_pass_kernel = low_pass.view(1, 1, 1, 2)
    high_pass_kernel = high_pass.view(1, 1, 1, 2)

    # 对称填充
    gray_image = F.pad(gray_image, (1, 1, 1, 1), mode="reflect")

    # 水平滤波（行处理）
    low_horiz = F.conv2d(gray_image, low_pass_kernel, stride=2, padding=0)
    high_horiz = F.conv2d(gray_image, high_pass_kernel, stride=2, padding=0)

    # 垂直滤波（列处理）
    low_pass_kernel = low_pass.view(1, 1, 2, 1)
    high_pass_kernel = high_pass.view(1, 1, 2, 1)

    LL = F.conv2d(low_horiz, low_pass_kernel, stride=2, padding=0)
    LH = F.conv2d(low_horiz, high_pass_kernel, stride=2, padding=0)
    HL = F.conv2d(high_horiz, low_pass_kernel, stride=2, padding=0)
    HH = F.conv2d(high_horiz, high_pass_kernel, stride=2, padding=0)

    # LH_upsampled = F.interpolate(LH, scale_factor=2, mode='nearest')  # 最近邻插值
    # HL_upsampled = F.interpolate(HL, scale_factor=2, mode='nearest')  # 最近邻插值
    # HH_upsampled = F.interpolate(HH, scale_factor=2, mode='nearest')  # 最近邻插值

    LH_upsampled = F.interpolate(LH, size=(H, W), mode='nearest')  # 最近邻插值
    HL_upsampled = F.interpolate(HL, size=(H, W), mode='nearest')  # 最近邻插值
    HH_upsampled = F.interpolate(HH, size=(H, W), mode='nearest')  # 最近邻插值

    return LL, (LH, HL, HH), (LH_upsampled, HL_upsampled, HH_upsampled)


    # LH_up = F.conv2d(low_horiz, high_pass_kernel, stride=1, padding=0)  # 水平高频子带
    # HL_up = F.conv2d(high_horiz, low_pass_kernel, stride=1, padding=0)  # 垂直高频子带
    # HH_up = F.conv2d(high_horiz, high_pass_kernel, stride=1, padding=0)  # 对角高频子带

    # return LL, (LH, HL, HH), (LH_up, HL_up, HH_up)


def haar_wavelet_2d_extra(image_tensor):
    """
    使用 Haar 小波变换分解 2D 图像，与 PyWavelets 的输出一致
    Args:
        image_tensor (torch.Tensor): 输入图像张量，形状为 [B, C, H, W]

    Returns:
        tuple: (低频分量 LL, (水平高频分量 LH, 垂直高频分量 HL, 对角高频分量 HH))
    """
    # 转为灰度图（PyWavelets 默认对灰度图进行处理）
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(image_tensor.device)
    gray_image = torch.sum(image_tensor * weights, dim=1, keepdim=True)  # 转为灰度图 [B, 1, H, W]

    # Haar 小波滤波器
    low_pass = torch.tensor([1 / torch.sqrt(torch.tensor(2.0)), 1 / torch.sqrt(torch.tensor(2.0))]).to(
        image_tensor.device)
    high_pass = torch.tensor([1 / torch.sqrt(torch.tensor(2.0)), -1 / torch.sqrt(torch.tensor(2.0))]).to(
        image_tensor.device)

    # 1D 滤波器扩展为 2D（分别作用于行和列）
    low_pass_kernel = low_pass.view(1, 1, 1, 2)  # 水平低频
    high_pass_kernel = high_pass.view(1, 1, 1, 2)  # 水平高频

    # 对称填充，模拟 pywt 的 `symmetric` 模式
    gray_image = F.pad(gray_image, (1, 1, 1, 1), mode="reflect")

    # 水平滤波（行处理）
    low_horiz = F.conv2d(gray_image, low_pass_kernel, stride=2, padding=0)
    high_horiz = F.conv2d(gray_image, high_pass_kernel, stride=2, padding=0)

    # 垂直滤波（列处理）
    low_pass_kernel = low_pass.view(1, 1, 2, 1)  # 垂直低频
    high_pass_kernel = high_pass.view(1, 1, 2, 1)  # 垂直高频

    LL = F.conv2d(low_horiz, low_pass_kernel, stride=2, padding=0)  # 低频子带
    LH = F.conv2d(low_horiz, high_pass_kernel, stride=2, padding=0)  # 水平高频子带
    HL = F.conv2d(high_horiz, low_pass_kernel, stride=2, padding=0)  # 垂直高频子带
    HH = F.conv2d(high_horiz, high_pass_kernel, stride=2, padding=0)  # 对角高频子带

    LH_up = F.conv2d(low_horiz, high_pass_kernel, stride=1, padding=0)  # 水平高频子带
    HL_up = F.conv2d(high_horiz, low_pass_kernel, stride=1, padding=0)  # 垂直高频子带
    HH_up = F.conv2d(high_horiz, high_pass_kernel, stride=1, padding=0)  # 对角高频子带

    return LL, (LH, HL, HH), (LH_up, HL_up, HH_up)


def ncc_loss(img1, img2, window_size=5):
    """
    Compute NCC loss between two images.
    Args:
        img1: Tensor of shape (B, C, H, W)
        img2: Tensor of shape (B, C, H, W)
        window_size: Size of the local window for NCC
    Returns:
        NCC loss: A scalar tensor
    """
    # Define padding to maintain the same shape
    padding = window_size // 2

    # Compute local mean
    mean1 = F.avg_pool2d(img1, kernel_size=window_size, stride=1, padding=padding)
    mean2 = F.avg_pool2d(img2, kernel_size=window_size, stride=1, padding=padding)

    # Compute zero-mean images
    zero_mean1 = img1 - mean1
    zero_mean2 = img2 - mean2

    # Compute local variance
    std1 = torch.sqrt(F.avg_pool2d(zero_mean1 ** 2, kernel_size=window_size, stride=1, padding=padding) + 1e-5)
    std2 = torch.sqrt(F.avg_pool2d(zero_mean2 ** 2, kernel_size=window_size, stride=1, padding=padding) + 1e-5)

    # Compute normalized cross-correlation
    ncc = F.avg_pool2d(zero_mean1 * zero_mean2, kernel_size=window_size, stride=1, padding=padding) / (
                std1 * std2 + 1e-5)

    # Convert NCC to loss
    ncc_loss = 1 - ncc.mean()  # Mean over batch and spatial dimensions
    return ncc_loss

def gaussian_blur(img, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to an image."""
    channels = img.shape[1]
    kernel = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    kernel = torch.exp(-0.5 * (kernel / sigma)**2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1) * kernel.view(1, -1, 1)
    kernel = kernel.repeat(channels, 1, 1, 1).cuda()
    return F.conv2d(img, kernel, padding=kernel_size // 2, groups=channels)

def downsample(img):
    """Downsample the image by a factor of 2."""
    return F.avg_pool2d(img, kernel_size=2, stride=2)

def upsample(img, size):
    """Upsample the image to the given size."""
    return F.interpolate(img, size=size, mode='bilinear', align_corners=False)

def laplacian_pyramid(img, levels=3):
    """Construct a Laplacian pyramid from an image."""
    gaussian_pyramid = [img]
    for _ in range(levels):
        blurred = gaussian_blur(gaussian_pyramid[-1])
        downsampled = downsample(blurred)
        gaussian_pyramid.append(downsampled)

    laplacian_pyramid = []
    for i in range(levels):
        upsampled = upsample(gaussian_pyramid[i + 1], size=gaussian_pyramid[i].shape[-2:])
        laplacian = gaussian_pyramid[i] - upsampled
        laplacian_pyramid.append(laplacian)

    laplacian_pyramid.append(gaussian_pyramid[-1])  # Add the top-level Gaussian
    return laplacian_pyramid

def laplacian_loss(img1, img2, levels=3):
    """Compute the Laplacian pyramid loss between two images."""
    laplacian1 = laplacian_pyramid(img1.unsqueeze(0), levels)
    laplacian2 = laplacian_pyramid(img2.unsqueeze(0), levels)
    # assert len(laplacian1) == 1, f"shape of img {img1.shape} and img2 {img2.shape}"
    loss = 0
    for l1, l2 in zip(laplacian1, laplacian2):
        loss += F.l1_loss(l1, l2)  # Use L1 loss for residuals
        # loss += 0.2 * ncc_loss(l1, l2)
    return loss



def penalize_outside_range(tensor, lower_bound=0.0, upper_bound=1.0):
    error = 0
    below_lower_bound = tensor[tensor < lower_bound]
    above_upper_bound = tensor[tensor > upper_bound]
    if below_lower_bound.numel():
        error += torch.mean((below_lower_bound - lower_bound) ** 2)
    if above_upper_bound.numel():
        error += torch.mean((above_upper_bound - upper_bound) ** 2)
    return error

def compute_sh_gauss_losses(shs_gauss, normals, N_samples=25):
    normal_vectors_expand = normals
    shs_gauss_expand = shs_gauss
    # normal_vectors_expand = normals.repeat_interleave(N_samples, dim=0)
    # shs_gauss_expand = shs_gauss.repeat_interleave(N_samples, dim=0)


    view_dir_unnorm = torch.empty(shs_gauss_expand.shape[0], 3, device=shs_gauss_expand.device).uniform_(-1.0, 1.0)
    view_dir = view_dir_unnorm
    # view_dir = view_dir_unnorm / (view_dir_unnorm.norm(dim=1, keepdim=True) + 1e-6)

    # eval SH_gauss for each sampled dir
    sampled_shs_gauss = eval_sh_point(view_dir, shs_gauss_expand[:, 0:1,
                                                :])  # we only have 0th sh channel for gaussian - shadow, no color bleeding


    # compute difference for shadowed and unshadowed values
    dot_product_n_v_raw = torch.sum(view_dir * normal_vectors_expand, dim=1, keepdim=True)
    dot_product_n_v = torch.clamp(dot_product_n_v_raw, min=0)
    shadowed_unshadowed_diff = sampled_shs_gauss - dot_product_n_v

    # SHgauss has to be in range 0-1, eq. 12
    sh_gauss_loss = penalize_outside_range(sampled_shs_gauss.view(-1), 0.0, 1.0)

    # Shadowed SH and unshadowed values need to be consistent, eq.14: ||eval_sh(view_dir) - max(0, dot(view_dir, normal))||^2
    consistency_loss = torch.mean(shadowed_unshadowed_diff ** 2)

    # Enforce shadowed SH to have lower values than unshadowed, eq. 15
    shadow_loss = (torch.clamp(shadowed_unshadowed_diff, min=0.0) ** 2).mean()

    return sh_gauss_loss, consistency_loss, shadow_loss


def compute_sh_env_loss(sh_env, N_samples=10):
    shs_view = sh_env.unsqueeze(0).permute(0, 2, 1).repeat(N_samples, 1, 1)
    view_dir_unnorm = torch.empty(shs_view.shape[0], 3, device=shs_view.device).uniform_(-1, 1)
    view_dir = view_dir_unnorm / view_dir_unnorm.norm(dim=1, keepdim=True)
    sampled_shs_gauss = eval_sh_point(view_dir, shs_view)

    # SH env has to be in R+, eq. 13
    sh_env_loss = penalize_outside_range(sampled_shs_gauss.view(-1), 0.0, torch.inf)

    return sh_env_loss


import pywt

def process_env_light(image: torch.Tensor, wavelet: str = 'haar', level: int = 6):
    """
    将输入 RGB 图像转换为灰度图，进行小波分解后展平并通过全连接层映射到指定形状。

    Args:
        image (torch.Tensor): 输入的 RGB 图像，形状为 [C, H, W]，C=3。
        wavelet (str): 小波分解使用的基函数，默认 'haar'。
        level (int): 小波分解的级别，默认 2。
        output_dim (tuple): 输出的形状，默认 (3, 9)。

    Returns:
        torch.Tensor: 映射后的嵌入张量，形状为 output_dim。
    """
    # 1. 将 RGB 图像转换为灰度图
    assert image.shape[0] == 3, "输入图像应为 RGB 格式 (3 通道)"
    gray_image = torch.mean(image, dim=0).cpu().numpy()  # 转为灰度图，形状为 [H, W]

    # 2. 小波分解
    coeffs = pywt.wavedec2(gray_image, wavelet, level=level)  # 多级分解
    flat_coeffs = []
    for c in coeffs:
        if isinstance(c, tuple):  # 高频部分 (LH, HL, HH)
            break
            # for subband in c:
            #     flat_coeffs.append(subband.flatten())
        else:  # 低频部分
            flat_coeffs.append(c.flatten())
    flat_coeffs = torch.tensor(torch.cat([torch.tensor(fc) for fc in flat_coeffs]),
                               dtype=torch.float32,
                               device=image.device)  # 展平

    return flat_coeffs
