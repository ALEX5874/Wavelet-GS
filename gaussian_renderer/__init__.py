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
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
# from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
# from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
# from pytorch3d.transforms import quaternion_to_matrix
from utils.normal_utils import compute_normal_world_space
from utils.sh_utils import eval_sh_shadowed, eval_sh_point, eval_sh_hemisphere, SH2RGB, RGB2SH


def generate_neural_gaussians(viewpoint_camera, pc: GaussianModel, visible_mask=None, is_training=False,
                              emb_idx=None, emb_env=None, unshadowed_image_loss_lambda=0, shadowed_image_loss_lambda=0):
    # wavelet-gaussian + lumi的方法
    ## view frustum filtering for acceleration
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

    feat = pc._anchor_feat[visible_mask]
    fres = pc._drifts_feat[visible_mask]

    drifts = pc.get_drifts[visible_mask]
    albedo = pc.get_albedo[visible_mask]

    valid_mask = (drifts.abs().mean(dim=-1) > 0.25)

    drifts = drifts[valid_mask]
    albedo = albedo[valid_mask]
    fres = fres[valid_mask]

    # 计算每对点之间的欧几里得距离
    # distances = torch.cdist(drifts, drifts, p=5)  # [N, N]，距离矩阵
    # # 设置邻域的距离阈值 r
    # r = 0.5  # 半径
    # neighbor_mask = distances < r  # [N, N]，邻域布尔矩阵


    anchor = pc.get_anchor[visible_mask]
    # albedo = pc.get_albedo[visible_mask]

    # anchor = anchor + drifts

    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    grid_scalingd = grid_scaling[valid_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
               feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
               feat[:, ::1, :1] * bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)  # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]

    res_local_view = torch.cat([fres, ob_view[valid_mask], ob_dist[valid_mask]], dim=1)  # [N, c+3+1]
    res_local_view_wodist = torch.cat([fres, ob_view[valid_mask]], dim=1)  # [N, c+3]

    # res_mask = torch.ones((drifts.shape[0], drifts.shape[0])).to(drifts.device)
    # res_local_view = pc.get_aggregate(drifts, res_local_view) # [N, c+3+1]

    # res_local_view_wodist = pc.get_aggregate(drifts, res_local_view_wodist)  # [N, c+3]

    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:, 0], dtype=torch.long,
                                          device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        anchor_opacity = pc.get_opacity_mlp(cat_local_view)  # [N, k]
        drifts_opacity = pc.get_opacity_del(res_local_view)  # [N, k]
        neural_opacity = pc.get_opacity_res(anchor_opacity, drifts_opacity)  # [N, k]
    else:
        anchor_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
        drifts_opacity = pc.get_opacity_del(res_local_view_wodist)
        neural_opacity = pc.get_opacity_res(anchor_opacity, drifts_opacity)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)

    drifts_opacity = drifts_opacity.reshape([-1, 1])
    maskd = (drifts_opacity > 0.0)
    maskd = maskd.view(-1)

    # select opacity
    opacity = neural_opacity[mask]

    opacityd = drifts_opacity[maskd]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            colora = pc.get_color_mlp(cat_local_view)
            colord = pc.get_color_del(res_local_view)
            color = pc.get_color_res(colora, colord)
        else:
            colora = pc.get_color_mlp(cat_local_view_wodist)
            colord = pc.get_color_del(res_local_view_wodist)
            color = pc.get_color_res(colora, colord)
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [mask]
    colord = colord.reshape([drifts.shape[0] * pc.d_offsets, 3])

    # get offset's cov
    if pc.add_cov_dist:
        scale_rota = pc.get_cov_mlp(cat_local_view)
        scale_rotd = pc.get_cov_del(res_local_view)
        scale_rot = pc.get_cov_res(scale_rota, scale_rotd)
    else:
        scale_rota = pc.get_cov_mlp(cat_local_view_wodist)
        scale_rotd = pc.get_cov_del(res_local_view_wodist)
        scale_rot = pc.get_cov_res(scale_rota, scale_rotd)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [mask]
    scale_rotd = scale_rotd.reshape([drifts.shape[0] * pc.d_offsets, 7])  # [mask]

    # offsets
    offsets = grid_offsets.view([-1, 3])  # [mask]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])  # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:, 3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets

    # combine for parallel masking
    concatenatedd = torch.cat([grid_scalingd, anchor[valid_mask], drifts], dim=-1)
    concatenated_repeatedd = repeat(concatenatedd, 'n (c) -> (n k) (c)', k=pc.d_offsets)

    concatenated_alld = torch.cat([concatenated_repeatedd, colord, scale_rotd], dim=-1)
    maskedd = concatenated_alld[maskd]
    scaling_repeatd, repeat_anchord, repeat_drifts, colord, scale_rotd = maskedd.split([6, 3, 3, 3, 7], dim=-1)

    # post-process cov
    scalingd = scaling_repeatd[:, 3:] * torch.sigmoid(scale_rotd[:, :3])  # * (1+torch.sigmoid(repeat_dist))
    rotd = pc.rotation_activation(scale_rotd[:, 3:7])

    # post-process offsets to get centers for gaussians
    offsetsd = repeat_drifts * scaling_repeatd[:, :3] * 0.5
    # repeat_anchord = repeat(anchor[valid_mask], 'n (c) -> (n k) (c)', k=pc.d_offsets)
    xyzd = repeat_anchord + offsetsd


    # calculate lumi override color
    # normal_vectors, multiplier = compute_normal_world_space(rot, scaling,
    #                                                         viewpoint_camera.world_view_transform,
    #                                                         xyz)
    normal_vectors, multiplier = compute_normal_world_space(rotd, scalingd,
                                                            viewpoint_camera.world_view_transform,
                                                            xyzd)
    sh_env = pc.compute_env_sh(emb_idx, emb_env)
    sh_random_noise = torch.randn_like(sh_env) * 0.025
    albedo = pc.get_albedo_mlp(albedo)[maskd]
    if unshadowed_image_loss_lambda > 0:
        rgb_precomp_unshadowed, _, shs_gauss_stacked, normal_vectors_stacked= compute_gaussian_rgb(albedo, res_local_view_wodist, pc, maskd,
                                                         # sh_env,
                                                         sh_env + sh_random_noise,
                                                         shadowed=False,
                                                         normal_vectors=normal_vectors)
    else:
        rgb_precomp_unshadowed = None


    if shadowed_image_loss_lambda > 0:
        rgb_precomp_shadowed, _, shs_gauss_stacked, normal_vectors_stacked = compute_gaussian_rgb(albedo, res_local_view_wodist, pc, maskd,
                                                       # sh_env,
                                                       sh_env + sh_random_noise,
                                                       normal_vectors=normal_vectors,
                                                       multiplier=multiplier)
    else:
        rgb_precomp_shadowed = None


    # if unshadowed_image_loss_lambda > 0 and shadowed_image_loss_lambda > 0:
    #     override_color = torch.sigmoid((rgb_precomp_unshadowed + rgb_precomp_shadowed) / 2)
    # elif unshadowed_image_loss_lambda > 0:
    #     override_color = torch.sigmoid(rgb_precomp_unshadowed)
    # elif shadowed_image_loss_lambda > 0:
    #     override_color = torch.sigmoid(rgb_precomp_shadowed)
    # else:
    #     override_color = None

    if shadowed_image_loss_lambda > 0:
        override_color = torch.sigmoid(rgb_precomp_shadowed)
    elif unshadowed_image_loss_lambda > 0:
        override_color = torch.sigmoid(rgb_precomp_unshadowed)
    else:
        override_color = None
        shs_gauss_stacked = None
        normal_vectors_stacked = None


    if override_color is not None:
        # colord = torch.sigmoid(colord + override_color)
        colord = torch.sigmoid(0.5 * colord + 0.5 * override_color)



    xyz = torch.cat([xyz, xyzd], dim=0)
    color = torch.cat([color, colord], dim=0)
    opacity = torch.cat([opacity, opacityd], dim=0)
    scaling = torch.cat([scaling, scalingd], dim=0)
    rot = torch.cat([rot, rotd], dim=0)
    neural_opacity = torch.cat([neural_opacity, drifts_opacity], dim=0)
    mask = torch.cat([mask, maskd], dim=0)


    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, shs_gauss_stacked, normal_vectors_stacked
    else:
        return xyz, color, opacity, scaling, rot




def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, visible_mask=None,
           retain_grad=False, emb_idx=None, emb_env=None, unshadowed_image_loss_lambda=0, shadowed_image_loss_lambda=0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, shs_gauss_stacked, normal_vectors_stacked  = generate_neural_gaussians(viewpoint_camera, pc,
                                                                                            visible_mask,
                                                                                            is_training=is_training,
                                                                                            emb_idx=emb_idx,
                                                                                            emb_env=emb_env,
                                                                                            unshadowed_image_loss_lambda=unshadowed_image_loss_lambda,
                                                                                            shadowed_image_loss_lambda=shadowed_image_loss_lambda)
    else:
        xyz, color, opacity, scaling, rot  = generate_neural_gaussians(viewpoint_camera, pc, visible_mask,
                                                                      is_training=is_training,
                                                                      emb_idx = emb_idx,
                                                                      emb_env=emb_env,
                                                                      unshadowed_image_loss_lambda = unshadowed_image_loss_lambda,
                                                                      shadowed_image_loss_lambda = shadowed_image_loss_lambda
                                                                      )



    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    # screenspace_points_abs = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
            # screenspace_points_abs.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "shs_gauss_stacked": shs_gauss_stacked,
                "normal_vectors_stacked": normal_vectors_stacked
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0


def compute_gaussian_rgb(albedo, drifts_feat, pc: GaussianModel, maskd, sh_scene, shadowed=True, multiplier=None, normal_vectors=None, env_hemisphere_lightning=True):
    #Computation of RGB could be implemented in CUDA. If you need it, please take care of it yourself.
    max_sh_degree = 2
    assert shadowed or (not shadowed and torch.is_tensor(normal_vectors))
    assert not shadowed or (shadowed and torch.is_tensor(multiplier))
    assert sh_scene.shape[-1] == (max_sh_degree+1)**2
    # assert torch.any(normal_vectors) == False

    features_positive, features_negative = pc.get_lumi_sh(drifts_feat)

    mask = (multiplier == 1)
    mask = torch.tensor(mask).to(albedo.device)
    # Select features based on the mask
    features = torch.where(mask.view(-1, 1, 1), features_positive[maskd], features_negative[maskd])
    # features = features[maskd]

    # shs_gauss = features.transpose(1, 2).view(-1, 3, (max_sh_degree + 1) ** 2)
    shs_gauss = features.transpose(1, 2).reshape(-1, 3, (max_sh_degree + 1) ** 2)
    if shadowed:
        # shs_gauss = features.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
        sh2intensity = eval_sh_shadowed(shs_gauss[:,0:1,:], sh_scene) #shs_gauss[:,0:1,:] for no color bleeding
    else:
        if env_hemisphere_lightning:
            sh2intensity = eval_sh_hemisphere(normal_vectors, sh_scene) #Expected output [B, 3]. Normals have to be unit.
        else:
            sh2intensity = eval_sh_point(normal_vectors, sh_scene) #Expected output [B, 3]. Normals have to be unit.

    intensity_hdr = torch.nn.functional.relu(sh2intensity)
    intensity = intensity_hdr**(1 / 2.2)  # linear to srgb
    # rgb = torch.clamp(intensity*albedo, 0.0)
    rgb = intensity * albedo

    # shs_gauss = features.transpose(1, 2).view(-1, 3, (max_sh_degree + 1) ** 2)
    # shs_gauss_stacked = torch.cat([shs_gauss], dim=0)
    # normal_vectors_stacked = torch.cat([normal_vectors], dim=0)

    return rgb, intensity, shs_gauss, normal_vectors