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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import math
import os
import torch.nn.functional as F
import pywt


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, image_path,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.nearest_id = []
        self.nearest_names = []
        self.neighbor_cam = []
        self.train_num = 0
        self.weights = 0

        # self.pixel_wavelet_num = self.get_pixel_num()

        self.mono_depth_name = image_name + '_aligned.npy'
        self.mono_depth_path = os.path.dirname(os.path.dirname(image_path)) + '/' + 'mono_depth' + '/'

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image = image
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # Compute the intrinsic matrix K
        self.K = self.get_intrinsic_matrix()

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix

    def get_depth(self, target_H, target_W):
        depth = np.load(self.mono_depth_path + self.mono_depth_name)
        depth_tensor = torch.from_numpy(depth).permute(2, 0, 1)  # 转换为 (C, H, W)
        depth_resized = F.interpolate(depth_tensor.unsqueeze(0), size=(target_H, target_W), mode="bilinear",
                                      align_corners=False)
        depth_resized = depth_resized.squeeze(0).cuda()  # 去掉 batch 维度
        # depth_resized = depth_resized.permute(1, 2, 0).squeeze(-1)
        # depth_resized = F.resize(depth_tensor, size=(target_H, target_W))
        # depth_resized = depth_resized.permute(1, 2, 0).to(self.data_device)
        return depth_resized

    def get_intrinsic_matrix(self):
        # Compute focal lengths from FoVx and FoVy
        f_x = self.image_width / (2 * np.tan(np.radians(self.FoVx) / 2))
        f_y = self.image_height / (2 * np.tan(np.radians(self.FoVy) / 2))

        # Principal point
        c_x = self.image_width / 2
        c_y = self.image_height / 2

        # Construct the intrinsic matrix K
        K = torch.tensor([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1]
        ], dtype=torch.float32)

        return K

    def get_direction(self):
        """
        计算相机的光轴方向 (默认相机坐标系的 Z 轴方向)。

        返回:
            torch.Tensor: 光轴方向向量 (3D)。
        """
        # 相机坐标系的 Z 轴方向
        z_axis = torch.tensor([0.0, 0.0, 1.0]).float()
        return torch.matmul(torch.tensor(self.R.T).float(), z_axis)  # 转到世界坐标系

    def get_pixel_num(self):
        image = self.original_image.clone().detach()
        wavelet = 'haar'
        level = 6


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
        return flat_coeffs.numel()

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

