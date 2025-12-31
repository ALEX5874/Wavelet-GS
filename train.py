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

import os
import numpy as np
import random
import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
#
# os.system('echo $CUDA_VISIBLE_DEVICES')
# os.environ['CUDA_VISIBLE_DEVICES']= "7"

import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, haar_wavelet_2d, laplacian_loss, get_img_grad_weight
from utils.loss_utils import compute_sh_env_loss, ncc_loss, process_env_light
from gaussian_renderer import generate_neural_gaussians, prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, update_lambdas
from utils.normal_utils import compute_normal_world_space
import cv2
# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


def compute_consistency_loss(output_view1, output_view2):
    color_view1 = output_view1[..., :3]
    color_view2 = output_view2[..., :3]

    loss = l1_loss(color_view1, color_view2)
    return loss


def get_mulitview_node(image_name):
    prefix, number = image_name.split('_')
    new_number = str(int(number) - 1).zfill(len(number))
    return f"{prefix}_{new_number}"


def get_4x4(R, T):
    """
    :param R: 3x3
    :param T: 3,
    :return:  4x4
    """
    full_matrix = np.eye(4)
    full_matrix[:3, :3] = R
    full_matrix[:3, 3] = T
    return full_matrix


def compute_transform(R1, T1, R2, T2):
    w2c1 = get_4x4(R1.transpose(), T1)
    w2c2 = get_4x4(R2.transpose(), T2)
    transformation_matrix = w2c2 @ np.linalg.inv(w2c2)
    return transformation_matrix


def warp_image(img, transform_matrix):
    img_array = np.array(img)

    height = img_array.shape[1]
    width = img_array.shape[2]
    img_array = np.transpose(img_array, (1, 2, 0))
    transform_matrix = transform_matrix.astype(np.float32)[:3, :3]

    warped_img = cv2.warpPerspective(img_array, transform_matrix, (width, height))
    warped_img = (warped_img * 255).astype(np.uint8)
    return warped_img


def overlay_images(img1, img2, alpha=0.5):
    img1 = np.array(img1)
    img2 = np.array(img2)
    img2 = (np.transpose(img2, (1, 2, 0)) * 255).astype(np.uint8)
    overlay = (img1 * alpha + img2 * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)


def show_image(image, window_name='Image'):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()
    ignorePatterns.append('benchmark_waymo_ours')
    ignorePatterns.append('benchmark_waymo_original')
    ignorePatterns.append('output_waymo')
    ignorePatterns.append('benchmark_360_ours')
    ignorePatterns.append('benchmark_360_ours_lumi')
    ignorePatterns.append('benchmark_tndt_ours')
    ignorePatterns.append('benchmark_snow_ours')
    ignorePatterns.append('benchmark_hkust_ours')
    ignorePatterns.append('benchmark_tndt_ours2')
    ignorePatterns.append('benchmark_snow_ours-1')
    ignorePatterns.append('benchmark_waymo_diffwave')

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')



def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    # use lumi-function

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, logger=logger)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # prepare look up table for appearance (lightning)
    viewpoint_stack = scene.getTrainCameras().copy()

    appearance_lut = {}
    for i, s in enumerate(viewpoint_stack):
        appearance_lut[s.image_name] = s.uid
    with open(os.path.join(scene.model_path, "appearance_lut.json"), "w") as outfile:
        json.dump(appearance_lut, outfile)



    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    lambda_consistency = 0.0
    images_dict = {}
    view_dict = {}


    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in scaffold-gs yet
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()


        cur_cam_id = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(cur_cam_id)
        image_name = viewpoint_cam.image_name
        loss_weight = 1.0


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if dataset.use_lumi_light:
            # Update loss lambdas depending on shadowed/unshadowed mode
            lambdas = update_lambdas(iteration, opt)
            shadowed_image_loss_lambda = lambdas["shadowed_image_loss_lambda"]
            unshadowed_image_loss_lambda = lambdas["unshadowed_image_loss_lambda"]
        else:
            shadowed_image_loss_lambda = 0
            unshadowed_image_loss_lambda = 0

        gt_image = viewpoint_cam.original_image.cuda()

        emb_idx = torch.ones((1, ), dtype=torch.long).cuda() * appearance_lut[viewpoint_cam.image_name]
        emb_env = process_env_light(gt_image)
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                            visible_mask=voxel_visible_mask,
                            retain_grad=retain_grad,
                            emb_idx=emb_idx,
                            emb_env=emb_env,
                            unshadowed_image_loss_lambda=unshadowed_image_loss_lambda,
                            shadowed_image_loss_lambda=shadowed_image_loss_lambda)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = (
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"],
            render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"])

        # gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg


        if iteration > opt.ncc_from_iter:
            nccloss = ncc_loss(gt_image, image)
            loss += opt.ncc_weight * nccloss

        # Wavelet loss
        # wait for implementation
        # if iteration > opt.wavelet_weight_from_iter:
        if opt.wavelet_weight > 0:
            if iteration <= opt.wavelet_loss_checkpoint1:
                wavelet_weight = opt.wavelet_weight * opt.wavelet_loss_scale1
            elif iteration > opt.wavelet_loss_checkpoint1 and iteration <= opt.wavelet_loss_checkpoint2:
                wavelet_weight = opt.wavelet_weight * opt.wavelet_loss_scale2
            else:
                wavelet_weight = opt.wavelet_weight * opt.wavelet_loss_scale3
            render_coeffs = haar_wavelet_2d(image.unsqueeze(0))
            gt_coeffs = haar_wavelet_2d(gt_image.unsqueeze(0))


            Ll2 = (
                l2_loss(render_coeffs[1][0], gt_coeffs[1][0]) +
                l2_loss(render_coeffs[1][1], gt_coeffs[1][1]) +
                l2_loss(render_coeffs[1][2], gt_coeffs[1][2])
            )

            Ll1 = l1_loss(render_coeffs[0], gt_coeffs[0])

            Ll1 = Ll1 + (
                l1_loss(render_coeffs[1][0], gt_coeffs[1][0]) +
                l1_loss(render_coeffs[1][1], gt_coeffs[1][1]) +
                l1_loss(render_coeffs[1][2], gt_coeffs[1][2])
            )

            lloss = laplacian_loss(gt_image, image)

            loss += wavelet_weight * (Ll1 + lloss + Ll2)


        if dataset.use_lumi_light:

            sh_env = gaussians.compute_env_sh(emb_idx, emb_env)

            # Update loss lambdas depending on shadowed/unshadowed mode
            lambdas = update_lambdas(iteration, opt)
            shadowed_image_loss_lambda = lambdas["shadowed_image_loss_lambda"]
            unshadowed_image_loss_lambda = lambdas["unshadowed_image_loss_lambda"]
            consistency_loss_lambda = lambdas["consistency_loss_lambda"]
            sh_gauss_lambda = lambdas["sh_gauss_lambda"]
            shadow_loss_lambda = lambdas["shadow_loss_lambda"]
            env_loss_lambda = lambdas["env_loss_lambda"]


            # photometric loss for unshadowed
            if unshadowed_image_loss_lambda > 0:

                image_unshadowed = render_pkg["render"]
                viewspace_point_tensor_unshadowed, visibility_filter, radii = render_pkg["viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"]


                Ll1_unshadowed = l1_loss(image_unshadowed, gt_image)
                unshadowed_image_loss = (1.0 - opt.lambda_dssim) * Ll1_unshadowed + opt.lambda_dssim * (
                            1.0 - ssim(image_unshadowed, gt_image))
                unshadowed_image_loss *= unshadowed_image_loss_lambda
            else:
                Ll1_unshadowed = torch.tensor(0)
                unshadowed_image_loss = torch.tensor(0)

            # photometric loss for shadowed
            if shadowed_image_loss_lambda > 0:

                image_shadowed = render_pkg["render"]
                viewspace_point_tensor_shadowed, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg[
                    "visibility_filter"], render_pkg["radii"]


                Ll1_shadowed = l1_loss(image_shadowed, gt_image)
                shadowed_image_loss = (1.0 - opt.lambda_dssim) * Ll1_shadowed + opt.lambda_dssim * (
                            1.0 - ssim(image_shadowed, gt_image))
                shadowed_image_loss *= shadowed_image_loss_lambda
            else:
                Ll1_shadowed = torch.tensor(0)
                shadowed_image_loss = torch.tensor(0)


            # Environment map loss for SH_env, eq. 13
            if env_loss_lambda > 0:
                sh_env_loss = env_loss_lambda * compute_sh_env_loss(sh_env)
            else:
                sh_env_loss = torch.tensor(0)


            lumi_loss = sh_env_loss
            loss += lumi_loss

        loss = loss_weight * loss

        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            images_dict[image_name] = image.cpu()
            view_dict[image_name] = viewpoint_cam.cpu()
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, opt, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, opt,
                    scene: Scene, renderFunc, renderArgs, wandb=None, logger=None):
    # lumi 版本
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss": Ll1, 'train_total_loss': loss, })

    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                appearance_lut = {}
                for i, s in enumerate(config['cameras']):
                    appearance_lut[s.image_name] = s.uid

                shadowed_image_loss_lambda = opt.shadowed_image_loss_lambda
                unshadowed_image_loss_lambda = opt.unshadowed_image_loss_lambda

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    # emb_idx = appearance_lut[viewpoint.image_name]
                    emb_idx = torch.ones((1,), dtype=torch.long).cuda() * appearance_lut[viewpoint.image_name]
                    emb_env = process_env_light(gt_image)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs,
                                   visible_mask=voxel_visible_mask,
                                   emb_idx=emb_idx,
                                   emb_env=emb_env,
                                   unshadowed_image_loss_lambda=unshadowed_image_loss_lambda,
                                   shadowed_image_loss_lambda=shadowed_image_loss_lambda
                                   )["render"],
                        0.0, 1.0)
                    # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(
                            f'{dataset_name}/' + config['name'] + "_view_{}/render".format(viewpoint.image_name),
                            image[None], global_step=iteration)
                        tb_writer.add_images(
                            f'{dataset_name}/' + config['name'] + "_view_{}/errormap".format(viewpoint.image_name),
                            (gt_image[None] - image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None] - image[None]).abs())

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/' + config['name'] + "_view_{}/ground_truth".format(
                                viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                logger.info(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/' + config['name'] + '/loss_viewpoint - l1_loss', l1_test,
                                         iteration)
                    tb_writer.add_scalar(f'{dataset_name}/' + config['name'] + '/loss_viewpoint - psnr', psnr_test,
                                         iteration)
                if wandb is not None:
                    wandb.log(
                        {f"{config['name']}_loss_viewpoint_l1_loss": l1_test, f"{config['name']}_PSNR": psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/' + 'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()


def render_set(model_path, name, iteration, views, gaussians, pipeline, opt, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}

    appearance_lut = {}
    for i, s in enumerate(views):
        appearance_lut[s.image_name] = s.uid

    # lambdas = update_lambdas_render()
    shadowed_image_loss_lambda = opt.shadowed_image_loss_lambda
    unshadowed_image_loss_lambda = opt.unshadowed_image_loss_lambda

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        gt = view.original_image[0:3, :, :]
        emb_idx = torch.ones((1,), dtype=torch.long).cuda() * appearance_lut[view.image_name]
        emb_env = process_env_light(gt)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background,
                            visible_mask=voxel_visible_mask,
                            emb_idx=emb_idx,
                            emb_env=emb_env,
                            unshadowed_image_loss_lambda=unshadowed_image_loss_lambda,
                            shadowed_image_loss_lambda=shadowed_image_loss_lambda
                            )
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        # gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, opt: OptimizationParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, opt, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, opt, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(5000, args.iterations + 1, 5000)]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != args.iterations:
        args.test_iterations.append(args.iterations)
    print(args.test_iterations)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')



    # try:
    #     saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    # except:
    #     logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    # if args.warmup:
    #     logger.info("\n Warmup finished! Reboot from last checkpoints")
    #     new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
    #     training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), op.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")

