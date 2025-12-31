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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
from scipy.spatial import cKDTree

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    # print(f'gt_image: {gt_image.shape}')
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def compute_camera_similarity(camera_list, top_n=5, max_angle=30, min_distance=0.5, max_distance=10):
    """
    根据 Camera 类计算视角相似性，并为每个相机找出相似的前 N 个相机。

    参数:
        camera_list (list): 包含 Camera 实例的列表。
        top_n (int): 每个相机选取的相似相机数量。
        max_angle (float): 最大允许的光轴夹角（单位：度）。
        min_distance (float): 最小允许的相机中心距离。
        max_distance (float): 最大允许的相机中心距离。

    返回:
        None: 结果直接更新到 Camera 实例的 `nearest_id` 属性中。
    """
    # 提取相机中心和光轴方向
    camera_centers = torch.stack([cam.camera_center for cam in camera_list]).cpu()  # [N, 3]
    camera_directions = torch.stack([cam.get_direction() for cam in camera_list])  # [N, 3]
    camera_directions = torch.nn.functional.normalize(camera_directions, dim=-1)  # 归一化方向向量

    # 计算相机之间的距离矩阵
    dist_matrix = torch.norm(camera_centers[:, None] - camera_centers[None, :], dim=-1)  # [N, N]

    # 计算方向相似性 (余弦相似度)
    direction_similarity = torch.matmul(camera_directions, camera_directions.T)  # [N, N]
    angles = torch.arccos(torch.clamp(direction_similarity, -1.0, 1.0)) * 180 / np.pi  # 转为角度

    for i, cam in enumerate(camera_list):
        # 获取与当前相机的距离和角度
        distances = dist_matrix[i]  # 当前相机与其他相机的距离
        angle_differences = angles[i]  # 当前相机与其他相机的方向夹角

        # 应用距离和角度过滤
        valid_mask = (distances > min_distance) & (distances < max_distance) & (angle_differences < max_angle)
        valid_mask[i] = False  # 排除自身

        # 提取满足条件的相机索引
        valid_indices = torch.where(valid_mask)[0]

        # 按方向相似性排序，其次按距离排序
        sorted_indices = valid_indices[torch.argsort(direction_similarity[i, valid_indices], descending=True)]
        sorted_indices = sorted_indices[:top_n]  # 仅保留前 top_n 个

        # 更新到 Camera 实例
        cam.nearest_id = [idx for idx in sorted_indices]
        # print(len(cam.nearest_id))
        # print(cam.nearest_id)


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # 仰角
    phi = np.arctan2(y, x)    # 方位角
    return r, theta, phi

# def calculate_camera_neighbors(camera_list, logger=None):
#     camera_centers = np.array([cam.camera_center.cpu().numpy() for cam in camera_list])
#
#     # 定义八个区域划分
#     def get_region(center, neighbor):
#         diff = neighbor - center
#         region = 0
#         if diff[2] >= 0:  # 前后
#             region += 1
#         if diff[0] >= 0:  # 左右
#             region += 2
#         if diff[1] >= 0:  # 上下
#             region += 4
#         return region + 1  # 区域编号从 1 开始
#
#     # 查找特定相机
#     def find_camera_by_name(name):
#         for cam in camera_list:
#             if cam.image_name == name:
#                 return cam
#         assert camera_list[0].image_name == name, f"{camera_list[0].image_name} is not {name}"
#
#     # cam_0_0 = find_camera_by_name('0_00000001')
#     # cam_0_1 = find_camera_by_name('0_00000002')
#     # cam_1_0 = find_camera_by_name('1_00000001')
#     # cam_2_0 = find_camera_by_name('2_00000001')
#     # cam_1_1 = find_camera_by_name('1_00000002')
#     # cam_2_1 = find_camera_by_name('2_00000002')
#     #
#     # if cam_0_0 and cam_1_0 and cam_2_0:
#     #     dist_0_1 = np.linalg.norm(cam_0_0.camera_center.cpu().numpy() - cam_1_0.camera_center.cpu().numpy())
#     #     dist_0_2 = np.linalg.norm(cam_0_0.camera_center.cpu().numpy() - cam_2_0.camera_center.cpu().numpy())
#     #     dist_1_2 = np.linalg.norm(cam_1_0.camera_center.cpu().numpy() - cam_2_0.camera_center.cpu().numpy())
#     #     dist001 = np.linalg.norm(cam_0_0.camera_center.cpu().numpy() - cam_0_1.camera_center.cpu().numpy())
#     #     dist101 = np.linalg.norm(cam_1_0.camera_center.cpu().numpy() - cam_1_1.camera_center.cpu().numpy())
#     #     dist201 = np.linalg.norm(cam_2_0.camera_center.cpu().numpy() - cam_2_1.camera_center.cpu().numpy())
#     #     distance_threshold = (dist_0_1 + dist_0_2) / 2
#     # else:
#     #     raise ValueError("Cameras with image_name '0-0', '1-0', or '2-0' not found.")
#
#     # assert dist_0_1 == dist_0_2, f"{dist_0_1} != {dist_0_2} and {dist_1_2} and {dist001} {dist101} {dist201}"
#
#     # 构建 KDTree 方便查找邻居
#     kdtree = cKDTree(camera_centers)
#     # 计算所有相机两两距离的最大值作为距离阈值
#     all_distances = kdtree.sparse_distance_matrix(kdtree, max_distance=np.inf).toarray()
#     max_distance_threshold = np.max(all_distances)
#     # assert max_distance_threshold == 1, f"{max_distance_threshold} != 1"
#
#     weights = 0
#     for i, center in enumerate(camera_centers):
#         # 查询所有其他相机并计算区域
#         distances, indices = kdtree.query(center, k=len(camera_centers))
#         regions = {region: [] for region in range(1, 9)}  # 初始化 8 个区域
#
#         for j, neighbor_idx in enumerate(indices[1:]):  # 排除自身
#             neighbor = camera_centers[neighbor_idx]
#             if distances[j + 1] <= max_distance_threshold / 10:  # 使用距离阈值判断是否为邻居
#                 region = get_region(center, neighbor)
#                 regions[region].append((neighbor_idx, distances[j + 1]))
#
#         # 对每个区域的邻居按距离排序
#         region_neighbors = {}
#         for region_id, neighbors in regions.items():
#             region_neighbors[region_id] = sorted(neighbors, key=lambda x: x[1])
#
#         # 计算权重
#         region_weight = np.zeros(8)
#         for region_id, neighbors in region_neighbors.items():
#             if neighbors:
#                 region_weight[region_id - 1] = len(neighbors)
#
#         # 计算总权重
#         active_regions = np.count_nonzero(region_weight)
#         total_weight = np.sum(region_weight) + np.exp(3 + active_regions)
#         # print(np.sum(region_weight))
#         # print(np.exp(3 + active_regions))
#         # assert np.sum(region_weight) == np.exp(3 + active_regions),f"{np.sum(region_weight)} == {np.exp(3 + active_regions)}"
#
#         # 将权重赋值给相机对象
#         camera_list[i].weights = total_weight
#         weights += total_weight
#
#     weights = weights / len(camera_list)
#     return weights
#

def calculate_camera_neighbors(camera_list, logger=None):
    for cam in camera_list:
        if '0_' in cam.image_name:
            cam.weights = 1
        else:
            cam.weights = 0