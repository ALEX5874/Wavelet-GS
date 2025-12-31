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
import torch

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from utils.loss_utils import process_env_light

def render_set(model_path, name, iteration, views, gaussians, pipeline, opt, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []

    appearance_lut = {}
    for i, s in enumerate(views):
        appearance_lut[s.image_name] = s.uid
    # lambdas = update_lambdas_render()
    # shadowed_image_loss_lambda = lambdas["shadowed_image_loss_lambda"]
    # unshadowed_image_loss_lambda = lambdas["unshadowed_image_loss_lambda"]
    shadowed_image_loss_lambda = opt.shadowed_image_loss_lambda
    unshadowed_image_loss_lambda = opt.unshadowed_image_loss_lambda

    emb_idx = []
    emb_env = []
    voxel_visible_mask = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        emb_idx1 = torch.ones((1,), dtype=torch.long).cuda() * appearance_lut[view.image_name]
        emb_env1 = process_env_light(gt).cuda()
        voxel_visible_mask1 = prefilter_voxel(view, gaussians, pipeline, background)
        emb_idx.append(emb_idx1)
        emb_env.append(emb_env1)
        voxel_visible_mask.append(voxel_visible_mask1)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize()
        gt = view.original_image[0:3, :, :].cuda()

        # emb_idx = torch.ones((1,), dtype=torch.long).cuda() * appearance_lut[view.image_name]
        # emb_env = process_env_light(gt).cuda()
        # voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background);

        t0 = time.time()

        render_pkg = render(view, gaussians, pipeline, background,
                            visible_mask=voxel_visible_mask[idx],
                            emb_idx=emb_idx[idx],
                            emb_env=emb_env[idx],
                            unshadowed_image_loss_lambda=unshadowed_image_loss_lambda,
                            shadowed_image_loss_lambda=shadowed_image_loss_lambda
                            )
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)

        rendering = render_pkg["render"]
        # gt = view.original_image[0:3, :, :]
        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, opt: OptimizationParams, skip_train : bool, skip_test : bool):
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
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, opt, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, opt, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), op.extract(args), args.skip_train, args.skip_test)
