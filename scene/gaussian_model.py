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
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from utils.sh_utils import eval_sh_shadowed, eval_sh_point, eval_sh_hemisphere, SH2RGB, RGB2SH
import cv2
from scene.wavelet_module import *
import time
import pywt
from einops import repeat
# from pytorch3d.transforms import quaternion_to_matrix
import random
import numpy as np
from random import randint


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int = 32,
                 n_offsets: int = 5,
                 voxel_size: float = 0.01,
                 update_depth: int = 3,
                 update_init_factor: int = 100,
                 update_hierachy_factor: int = 4,
                 use_feat_bank: bool = False,
                 appearance_dim: int = 32,
                 ratio: int = 1,
                 add_opacity_dist: bool = False,
                 add_cov_dist: bool = False,
                 add_color_dist: bool = False,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.active_sh_degree = 2
        self.max_sh_degree = 2

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self._drifts = torch.empty(0)
        self._drifts_feat = torch.empty(0)
        self.d_offsets = 5

        self._albedo = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3 + 1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.add_color_dist = add_color_dist
        self.color_dist_dim = 1 if add_color_dist else 0
        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.add_opacity_dist = add_opacity_dist
        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0

        self.mlp_opacity = mlp_opacity(feat_dim, add_opacity_dist).cuda()
        self.mlp_cov = mlp_cov(feat_dim, add_cov_dist).cuda()
        self.mlp_color = mlp_color(feat_dim, add_color_dist).cuda()

        self.del_opacity = del_opacity(feat_dim, add_opacity_dist, self.d_offsets).cuda()
        self.del_cov = del_cov(feat_dim, add_cov_dist, self.d_offsets).cuda()
        self.del_color = del_color(feat_dim, add_color_dist, self.d_offsets).cuda()

        self.res_opacity = res_opacity(feat_dim, n_offsets).cuda()
        self.res_cov = res_cov(feat_dim, n_offsets).cuda()
        self.res_color = res_color(feat_dim, n_offsets).cuda()

        self.lumi_sh = lumi_sh(feat_dim, self.max_sh_degree, add_color_dist, self.d_offsets).cuda()
        self.albedo_mlp = albedo_mlp(feat_dim, self.d_offsets).cuda()
        # self.env_sh = env_sh_mlp(feat_dim, self.max_sh_degree, add_color_dist, n_offsets).cuda()


    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()

        self.del_opacity.eval()
        self.del_cov.eval()
        self.del_color.eval()

        self.res_opacity.eval()
        self.res_cov.eval()
        self.res_color.eval()

        self.lumi_sh.eval()
        self.env_sh.eval()
        self.albedo_mlp.eval()

        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()

        self.del_opacity.train()
        self.del_cov.train()
        self.del_color.train()

        self.res_opacity.train()
        self.res_cov.train()
        self.res_color.train()

        self.lumi_sh.train()
        self.env_sh.train()
        self.albedo_mlp.train()

        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._albedo,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._anchor,
         self._offset,
         self._albedo,
         self._local,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    def set_envsh(self, num_cameras, pixel_num):
        self.env_sh = env_sh_mlp(N_vocab=num_cameras, pixel_num=pixel_num).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0 * self.scaling_activation(self._scaling)

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_opacity_del(self):
        return self.del_opacity

    @property
    def get_cov_del(self):
        return self.del_cov

    @property
    def get_color_del(self):
        return self.del_color

    @property
    def get_opacity_res(self):
        return self.res_opacity

    @property
    def get_cov_res(self):
        return self.res_cov

    @property
    def get_color_res(self):
        return self.res_color

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_drifts(self):
        return self._drifts

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor


    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_albedo(self):
        return torch.clamp(SH2RGB(self._albedo), 0.0)

    @property
    def get_albedo_mlp(self):
        return self.albedo_mlp

    @property
    def get_lumi_sh(self):
        return self.lumi_sh

    def get_features(self, multiplier):
        features_positive, features_negative = self.lumi_sh(self._drifts_feat)
        mask = multiplier == 1
        # Select features based on the mask
        features = torch.where(mask.view(-1, 1, 1), features_positive, features_negative)
        return features


    def compute_env_sh(self, emb_idx, emb_env):
        return self.env_sh(emb_idx, emb_env)


    def compute_gaussian_rgb(self, sh_scene, shadowed=True, multiplier=None, normal_vectors=None, env_hemisphere_lightning=True):
        #Computation of RGB could be implemented in CUDA. If you need it, please take care of it yourself.
        assert shadowed or (not shadowed and torch.is_tensor(normal_vectors))
        assert not shadowed or (shadowed and torch.is_tensor(multiplier))
        assert sh_scene.shape[-1] == (self.max_sh_degree+1)**2

        albedo = self.get_albedo
        if shadowed:
            shs_gauss = self.get_features(multiplier).transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
            sh2intensity = eval_sh_shadowed(shs_gauss[:,0:1,:], sh_scene) #shs_gauss[:,0:1,:] for no color bleeding
        else:
            if env_hemisphere_lightning:
                sh2intensity = eval_sh_hemisphere(normal_vectors, sh_scene) #Expected output [B, 3]. Normals have to be unit.
            else:
                sh2intensity = eval_sh_point(normal_vectors, sh_scene) #Expected output [B, 3]. Normals have to be unit.

        intensity_hdr = torch.nn.functional.relu(sh2intensity)
        intensity = intensity_hdr**(1 / 2.2)  # linear to srgb
        rgb = torch.clamp(intensity*albedo, 0.0)

        return rgb, intensity

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size

        return data

    def wavelet_decomposition_pointcloud(self, point_cloud, wavelet='haar', level=1):
        """
        对点云数据进行小波分解，将每个维度分解为高频和低频部分。

        参数：
        - point_cloud: numpy array, 点云数据，形状为 [N, 3]
        - wavelet: str, 小波类型（如 'haar', 'db1'）
        - level: int, 小波分解层数

        返回：
        - low_freq: numpy array, 低频部分，形状与输入相同
        - high_freq: numpy array, 高频部分，形状与输入相同
        """

        low_freq = np.zeros_like(point_cloud)
        high_freq = np.zeros_like(point_cloud)

        for i in range(3):
            coeffs = pywt.wavedec(point_cloud[:, i], wavelet, level=level)

            # Reconstruct low-frequency part (approximation)
            low_freq[:, i] = pywt.waverec([coeffs[0]] + [None] * level, wavelet)[:low_freq.shape[0]]

            # Reconstruct high-frequency part (details)
            high_freq[:, i] = pywt.waverec([None] * (level) + [coeffs[j] for j in range(1, level + 1)], wavelet)[
                              :low_freq.shape[0]]

        return low_freq, high_freq

    def wavelet_sample(self, data, wavelet_name="db2", levels=3):
        torch.cuda.synchronize(); t0 = time.time()
        self.levels = levels
        self.position = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda()

        decomposed = wavelet_decompose_pointcloud(data, wavelet_name, levels)

        app = decomposed['approximation']
        det = decomposed['details']

        for cur_level in range(self.levels + 1):
            if cur_level == 0:
                new_positions = app
                new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
                self.position = torch.concat((self.position, new_positions), dim=0)
                self._level = torch.concat((self._level, new_level), dim=0)
            else:
                new_positions = det[-cur_level]
                new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
                self.position = torch.concat((self.position, new_positions), dim=0)
                self._level = torch.concat((self._level, new_level), dim=0)


        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building wavelet time: {int(time_diff // 60)} min {time_diff % 60} sec")


    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, logger=None):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]  # points应该是[n, 3]
        # points = torch.tensor(pcd.points[::self.ratio]).float().cuda()

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')


        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        low_freq, high_freq = self.wavelet_decomposition_pointcloud(points)

        low_point_cloud = torch.tensor(np.asarray(low_freq)).float().cuda()
        high_point_cloud = torch.tensor(np.asarray(high_freq)).float().cuda()

        albedo = RGB2SH(torch.zeros_like(low_point_cloud).float().cuda())


        offsets = torch.zeros((low_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((low_point_cloud.shape[0], self.feat_dim)).float().cuda()
        driftss_feat = torch.zeros((high_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", low_point_cloud.shape[0])
        logger.info(f'Number of points at initialisation : {low_point_cloud.shape[0]}')

        print("Number of points at initialisation : ", high_point_cloud.shape[0])
        logger.info(f'Number of points at initialisation : {high_point_cloud.shape[0]}')
        logger.info(f'Mean : {torch.mean(high_point_cloud.abs(), dim=0)}')
        logger.info(f'MAX : {torch.max(high_point_cloud, dim=0)}')
        logger.info(f'MIN : {torch.min(high_point_cloud, dim=0)}')

        dist2 = torch.clamp_min(distCUDA2(low_point_cloud).float().cuda(), 0.0000001)  # 设置下限
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((low_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((low_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(low_point_cloud.requires_grad_(True))
        self._drifts = nn.Parameter(high_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._drifts_feat = nn.Parameter(driftss_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        self._albedo = nn.Parameter(albedo.requires_grad_(True))


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "anchor"},
                {'params': [self._drifts], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "drifts"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale,
                 "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._drifts_feat], 'lr': training_args.feature_lr, "name": "drifts_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': [self._albedo], 'lr': training_args.albedo_lr_init, "name": "albedo"},
                {'params': self.albedo_mlp.parameters(), 'lr': training_args.albedo_mlp_lr_init, "name": "albedo_mlp"},
                {'params': self.lumi_sh.parameters(), 'lr': training_args.lumi_sh_lr_init, "name": "lumi_sh"},
                {'params': self.env_sh.parameters(), 'lr': training_args.env_sh_lr_init, "name": "env_sh"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init,
                 "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init,
                 "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init,
                 "name": "embedding_appearance"},


                {'params': self.del_opacity.parameters(), 'lr': training_args.del_opacity_lr_init,
                 "name": "del_opacity"},
                {'params': self.del_cov.parameters(), 'lr': training_args.del_cov_lr_init, "name": "del_cov"},
                {'params': self.del_color.parameters(), 'lr': training_args.del_color_lr_init, "name": "del_color"},
                {'params': self.res_opacity.parameters(), 'lr': training_args.res_opacity_lr_init,
                 "name": "res_opacity"},
                {'params': self.res_cov.parameters(), 'lr': training_args.res_cov_lr_init, "name": "res_cov"},
                {'params': self.res_color.parameters(), 'lr': training_args.res_color_lr_init, "name": "res_color"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "anchor"},
                {'params': [self._drifts], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "drifts"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale,
                 "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._drifts_feat], 'lr': training_args.feature_lr, "name": "drifts_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': [self._albedo], 'lr': training_args.albedo_lr_init, "name": "albedo"},
                {'params': self.albedo_mlp.parameters(), 'lr': training_args.albedo_mlp_lr_init, "name": "albedo_mlp"},
                {'params': self.lumi_sh.parameters(), 'lr': training_args.lumi_sh_lr_init, "name": "lumi_sh"},
                {'params': self.env_sh.parameters(), 'lr': training_args.env_sh_lr_init, "name": "env_sh"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init,
                 "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init,
                 "name": "embedding_appearance"},


                {'params': self.del_opacity.parameters(), 'lr': training_args.del_opacity_lr_init,
                 "name": "del_opacity"},
                {'params': self.del_cov.parameters(), 'lr': training_args.del_cov_lr_init, "name": "del_cov"},
                {'params': self.del_color.parameters(), 'lr': training_args.del_color_lr_init, "name": "del_color"},
                {'params': self.res_opacity.parameters(), 'lr': training_args.res_opacity_lr_init,
                 "name": "res_opacity"},
                {'params': self.res_cov.parameters(), 'lr': training_args.res_cov_lr_init, "name": "res_cov"},
                {'params': self.res_color.parameters(), 'lr': training_args.res_color_lr_init, "name": "res_color"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "anchor"},
                {'params': [self._drifts], 'lr': training_args.drifts_lr_init * self.spatial_lr_scale,
                 "name": "drifts"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale,
                 "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._drifts_feat], 'lr': training_args.feature_lr, "name": "drifts_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': [self._albedo], 'lr': training_args.albedo_lr_init, "name": "albedo"},
                {'params': self.albedo_mlp.parameters(), 'lr': training_args.albedo_mlp_lr_init, "name": "albedo_mlp"},
                {'params': self.lumi_sh.parameters(), 'lr': training_args.lumi_sh_lr_init, "name": "lumi_sh"},
                {'params': self.env_sh.parameters(), 'lr': training_args.env_sh_lr_init, "name": "env_sh"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init,
                 "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},


                {'params': self.del_opacity.parameters(), 'lr': training_args.del_opacity_lr_init,
                 "name": "del_opacity"},
                {'params': self.del_cov.parameters(), 'lr': training_args.del_cov_lr_init, "name": "del_cov"},
                {'params': self.del_color.parameters(), 'lr': training_args.del_color_lr_init, "name": "del_color"},
                {'params': self.res_opacity.parameters(), 'lr': training_args.res_opacity_lr_init,
                 "name": "res_opacity"},
                {'params': self.res_cov.parameters(), 'lr': training_args.res_cov_lr_init, "name": "res_cov"},
                {'params': self.res_color.parameters(), 'lr': training_args.res_color_lr_init, "name": "res_color"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.position_lr_max_steps)
        self.drifts_scheduler_args = get_expon_lr_func(lr_init=training_args.drifts_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.drifts_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=training_args.drifts_lr_delay_mult,
                                                       max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.offset_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=training_args.offset_lr_delay_mult,
                                                       max_steps=training_args.offset_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                            lr_final=training_args.mlp_opacity_lr_final,
                                                            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                            max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                        lr_final=training_args.mlp_cov_lr_final,
                                                        lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                        max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                          lr_final=training_args.mlp_color_lr_final,
                                                          lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                          max_steps=training_args.mlp_color_lr_max_steps)
        self.del_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.del_opacity_lr_init,
                                                            lr_final=training_args.del_opacity_lr_final,
                                                            lr_delay_mult=training_args.del_opacity_lr_delay_mult,
                                                            max_steps=training_args.del_opacity_lr_max_steps)

        self.del_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.del_cov_lr_init,
                                                        lr_final=training_args.del_cov_lr_final,
                                                        lr_delay_mult=training_args.del_cov_lr_delay_mult,
                                                        max_steps=training_args.del_cov_lr_max_steps)

        self.del_color_scheduler_args = get_expon_lr_func(lr_init=training_args.del_color_lr_init,
                                                          lr_final=training_args.del_color_lr_final,
                                                          lr_delay_mult=training_args.del_color_lr_delay_mult,
                                                          max_steps=training_args.del_color_lr_max_steps)
        self.res_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.res_opacity_lr_init,
                                                            lr_final=training_args.res_opacity_lr_final,
                                                            lr_delay_mult=training_args.res_opacity_lr_delay_mult,
                                                            max_steps=training_args.res_opacity_lr_max_steps)

        self.res_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.res_cov_lr_init,
                                                        lr_final=training_args.res_cov_lr_final,
                                                        lr_delay_mult=training_args.res_cov_lr_delay_mult,
                                                        max_steps=training_args.res_cov_lr_max_steps)

        self.res_color_scheduler_args = get_expon_lr_func(lr_init=training_args.res_color_lr_init,
                                                          lr_final=training_args.res_color_lr_final,
                                                          lr_delay_mult=training_args.res_color_lr_delay_mult,
                                                          max_steps=training_args.res_color_lr_max_steps)

        self.lumi_sh_scheduler_args = get_expon_lr_func(lr_init=training_args.lumi_sh_lr_init,
                                                          lr_final=training_args.lumi_sh_lr_final,
                                                          lr_delay_mult=training_args.lumi_sh_lr_delay_mult,
                                                          max_steps=training_args.lumi_sh_lr_max_steps)

        self.env_sh_scheduler_args = get_expon_lr_func(lr_init=training_args.env_sh_lr_init,
                                                          lr_final=training_args.env_sh_lr_final,
                                                          lr_delay_mult=training_args.env_sh_lr_delay_mult,
                                                          max_steps=training_args.env_sh_lr_max_steps)

        self.albedo_scheduler_args = get_expon_lr_func(lr_init=training_args.albedo_lr_init,
                                                          lr_final=training_args.albedo_lr_final,
                                                          lr_delay_mult=training_args.albedo_lr_delay_mult,
                                                          max_steps=training_args.albedo_lr_max_steps)
        self.albedo_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.albedo_mlp_lr_init,
                                                          lr_final=training_args.albedo_mlp_lr_final,
                                                          lr_delay_mult=training_args.albedo_mlp_lr_delay_mult,
                                                          max_steps=training_args.albedo_mlp_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                                    lr_final=training_args.mlp_featurebank_lr_final,
                                                                    lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                                    max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                               lr_final=training_args.appearance_lr_final,
                                                               lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                               max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "drifts":
                lr = self.drifts_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "del_opacity":
                lr = self.del_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "del_cov":
                lr = self.del_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "del_color":
                lr = self.del_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "res_opacity":
                lr = self.res_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "res_cov":
                lr = self.res_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "res_color":
                lr = self.res_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "albedo":
                lr = self.albedo_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "albedo_mlp":
                lr = self.albedo_mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "lumi_sh":
                lr = self.lumi_sh_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "env_sh":
                lr = self.env_sh_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1] * self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        for i in range(self._drifts_feat.shape[1]):
            l.append('f_drifts_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._albedo.shape[1]):
            l.append('albedo_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        drifts = self._drifts.detach().cpu().numpy()
        # anchor = anchor + drifts
        # normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        drifts_feat = self._drifts_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, drifts, offset, anchor_feat, drifts_feat, opacities, scale, rotation, albedo), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                           np.asarray(plydata.elements[0]["y"]),
                           np.asarray(plydata.elements[0]["z"])), axis=1).astype(np.float32)
        drifts = np.stack((np.asarray(plydata.elements[0]["nx"]),
                           np.asarray(plydata.elements[0]["ny"]),
                           np.asarray(plydata.elements[0]["nz"])), axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key=lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        drifts_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_drifts_feat")]
        drifts_feat_names = sorted(drifts_feat_names, key=lambda x: int(x.split('_')[-1]))
        drifts_feats = np.zeros((drifts.shape[0], len(drifts_feat_names)))
        for idx, attr_name in enumerate(drifts_feat_names):
            drifts_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key=lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        albedo_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("albedo_")]
        albedo_names = sorted(albedo_names, key = lambda x: int(x.split('_')[-1]))
        albedo = np.zeros((anchor.shape[0], len(albedo_names)))
        for idx, attr_name in enumerate(albedo_names):
            albedo[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._anchor_feat = nn.Parameter(
            torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._drifts_feat = nn.Parameter(
            torch.tensor(drifts_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(
            torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._drifts = nn.Parameter(torch.tensor(drifts, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or \
                    'sh' in group['name'] or \
                    'del' in group['name'] or \
                    'res' in group['name'] or \
                    'conv' in group['name'] or \
                    'feat_base' in group['name'] or \
                    'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # statis grad information to guide liftting.
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask,
                        anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0

        temp_opacity = temp_opacity[:self.n_offsets * self.opacity_accum[anchor_visible_mask].shape[0]].view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)

        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask[:self.offset_gradient_accum[anchor_visible_mask].shape[0]]
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter[:combined_mask[temp_mask].shape[0]]

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm[:self.offset_gradient_accum[combined_mask].shape[0]]
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or \
                    'sh' in group['name'] or \
                    'del' in group['name'] or \
                    'res' in group['name'] or \
                    'conv' in group['name'] or \
                    'feat_base' in group['name'] or \
                    'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_anchor(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._drifts = optimizable_tensors["drifts"]
        self._drifts_feat = optimizable_tensors["drifts_feat"]
        self._albedo = optimizable_tensors["albedo"]

    def apply_drifts_mask(self, grads, grad_threshold):
        mask = (grads < grad_threshold)
        # rand_mask = torch.rand_like(mask.float()) > (0.5 ** (self.update_depth + 1))
        # rand_mask = rand_mask.cuda()
        # mask = torch.logical_and(mask, rand_mask)
        assert f"candidate_mask = (grads >= cur_threshold): {grads.shape} and {mask.shape}"
        self._drifts[mask] = self._drifts[mask] * 0.5
        self._drifts_feat[mask] = 0.0

        # grow_mask =

    def anchor_growing(self, grads, threshold, offset_mask):
        ##
        init_length = self.get_anchor.shape[0] * self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold * ((self.update_hierachy_factor // 2) ** i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            # random pick
            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (i + 1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            drifts = self.get_drifts.clone()
            #TMOD


            structure_mask = (torch.mean(drifts.abs(), dim=-1) >= 15.0)
            # structure_mask = structure_mask.unsqueeze(dim=1).repeat([1, self.n_offsets])
            structure_mask = structure_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view([-1])[:candidate_mask.shape[0]]
            # assert structure_mask.shape[0] == 1, f"the shape is {structure_mask.shape} and {rand_mask.shape}"
            structure_mask = torch.logical_and(structure_mask, rand_mask)

            rand_mask = torch.rand_like(structure_mask.float()) > 0.9
            rand_mask = rand_mask.cuda()
            structure_mask = torch.logical_and(structure_mask, rand_mask)

            candidate_mask = torch.logical_or(candidate_mask, structure_mask)

            length_inc = self.get_anchor.shape[0] * self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')],
                                           dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor ** i)
            cur_size = self.voxel_size * size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True,
                                                                        dim=0)

            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i * chunk_size:(
                                                                                                                                i + 1) * chunk_size,
                                                                                         :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size  # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(
                    0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_albedo = inverse_sigmoid(
                    0.1 * torch.ones((candidate_anchor.shape[0], 3), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[
                    candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][
                    remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat(
                    [1, self.n_offsets, 1]).float().cuda()

                new_drifts = 2 * torch.rand_like(candidate_anchor) - 1.0
                new_drifts_feat = torch.zeros([candidate_anchor.shape[0], self._drifts_feat.shape[1]], device='cuda')

                d = {
                    "anchor": candidate_anchor,
                    "drifts": new_drifts,
                    "drifts_feat": new_drifts_feat,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                    "albedo": new_albedo,
                }

                temp_anchor_demon = torch.cat(
                    [self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat(
                    [self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._drifts = optimizable_tensors["drifts"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._drifts_feat = optimizable_tensors["drifts_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                self._albedo = optimizable_tensors["albedo"]


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom  # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)

        # init_length = self.get_anchor.shape[0]
        # grads[~offset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (
                    torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        anchor_grads_norm = torch.norm(anchor_grads, dim=-1)

        # mask_threshold = grad_threshold * ((self.update_hierachy_factor // 2) ** (self.update_depth + 1))
        self.apply_drifts_mask(anchor_grads_norm, grad_threshold)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [self.get_anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity * self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval * success_threshold).squeeze(dim=1)  # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum() > 0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0] > 0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode='split', pixel_num=100):  # split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))

            opacity_del = torch.jit.trace(self.del_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_del.save(os.path.join(path, 'opacity_del.pt'))
            cov_del = torch.jit.trace(self.del_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_del.save(os.path.join(path, 'cov_del.pt'))
            color_del = torch.jit.trace(self.del_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_del.save(os.path.join(path, 'color_del.pt'))

            opacity_res = torch.jit.trace(self.res_opacity, (torch.rand(1, self.feat_dim).cuda(), torch.rand(1, self.feat_dim).cuda()))
            opacity_res.save(os.path.join(path, 'opacity_res.pt'))
            cov_res = torch.jit.trace(self.res_cov, (torch.rand(1, self.feat_dim).cuda(), torch.rand(1, self.feat_dim).cuda()))
            cov_res.save(os.path.join(path, 'cov_res.pt'))
            color_res = torch.jit.trace(self.res_color, (torch.rand(1, self.feat_dim).cuda(), torch.rand(1, self.feat_dim).cuda()))
            color_res.save(os.path.join(path, 'color_res.pt'))

            lumi_sh = torch.jit.trace(self.lumi_sh, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            lumi_sh.save(os.path.join(path, 'lumi_sh.pt'))
            env_sh = torch.jit.trace(self.env_sh, (torch.ones((1,), dtype=torch.long).cuda(), torch.rand(1, pixel_num).cuda()))
            env_sh.save(os.path.join(path, 'env_sh.pt'))
            albedo_mlp = torch.jit.trace(self.albedo_mlp, (torch.rand(1, 3).cuda()))
            albedo_mlp.save(os.path.join(path, 'albedo_mlp.pt'))


            if self.use_feat_bank:
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+self.level_dim).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
            if self.appearance_dim > 0:
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
            self.train()
        elif mode == 'unite':
            param_dict = {}
            param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
            param_dict['cov_mlp'] = self.mlp_cov.state_dict()
            param_dict['color_mlp'] = self.mlp_color.state_dict()

            param_dict['opacity_del'] = self.del_opacity.state_dict()
            param_dict['cov_del'] = self.del_cov.state_dict()
            param_dict['color_del'] = self.del_color.state_dict()

            param_dict['opacity_res'] = self.res_opacity.state_dict()
            param_dict['cov_res'] = self.res_cov.state_dict()
            param_dict['color_res'] = self.res_color.state_dict()

            if self.use_feat_bank:
                param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            if self.appearance_dim > 0:
                param_dict['appearance'] = self.embedding_appearance.state_dict()
            torch.save(param_dict, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError

    def load_mlp_checkpoints(self, path, mode='split'):  # split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()

            self.del_opacity = torch.jit.load(os.path.join(path, 'opacity_del.pt')).cuda()
            self.del_cov = torch.jit.load(os.path.join(path, 'cov_del.pt')).cuda()
            self.del_color = torch.jit.load(os.path.join(path, 'color_del.pt')).cuda()

            self.res_opacity = torch.jit.load(os.path.join(path, 'opacity_res.pt')).cuda()
            self.res_cov = torch.jit.load(os.path.join(path, 'cov_res.pt')).cuda()
            self.res_color = torch.jit.load(os.path.join(path, 'color_res.pt')).cuda()

            self.lumi_sh = torch.jit.load(os.path.join(path, 'lumi_sh.pt')).cuda()
            self.env_sh = torch.jit.load(os.path.join(path, 'env_sh.pt')).cuda()
            self.albedo_mlp = torch.jit.load(os.path.join(path, 'albedo_mlp.pt')).cuda()

            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])

            self.del_opacity.load_state_dict(checkpoint['opacity_del'])
            self.del_cov.load_state_dict(checkpoint['cov_del'])
            self.del_color.load_state_dict(checkpoint['color_del'])

            self.res_opacity.load_state_dict(checkpoint['opacity_res'])
            self.res_cov.load_state_dict(checkpoint['cov_res'])
            self.res_color.load_state_dict(checkpoint['color_res'])

            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError

