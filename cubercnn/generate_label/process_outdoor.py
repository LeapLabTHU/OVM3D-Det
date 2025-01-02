# Copyright (c) Meta Platforms, Inc. and affiliates
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from cubercnn import util
import math
from cubercnn.generate_label.util import *
from cubercnn.generate_label.raytrace import calc_dis_ray_tracing, calc_inside_ratio
from tqdm import tqdm


def create_uv_depth(depth, mask=None):
    """Generate UV-Depth point cloud."""
    if mask is not None:
        depth = depth * mask
    x, y = np.meshgrid(
        np.linspace(0, depth.shape[1] - 1, depth.shape[1]),
        np.linspace(0, depth.shape[0] - 1, depth.shape[0])
    )
    uv_depth = np.stack((x, y, depth), axis=-1)
    uv_depth = uv_depth.reshape(-1, 3)
    return uv_depth[uv_depth[:, 2] != 0]


def process_ground(info_ground, im_id, depth, input_folder, K):
    if im_id not in info_ground or not info_ground[im_id]:
        return False, None

    ground_mask = np.load(f'{input_folder}/ground_mask/{im_id}.npy')
    ground_mask = erode_mask(ground_mask.astype(float), 4, 4)

    ground_mask = ground_mask[np.argmax(info_ground[im_id]['conf'])]
    ground_depth = depth * ground_mask.squeeze()

    uv_depth = create_uv_depth(ground_depth)
    pseudo_lidar_ground = project_image_to_cam(uv_depth, np.array(K))

    # If the number of points is less than 10, the ground plane is not reliable
    if pseudo_lidar_ground.shape[0] > 10:
        ground_equ = extract_ground(pseudo_lidar_ground)
        return True, ground_equ
    return False, None


def process_instances(mask_instance, depth, K, info_i, cat_prior, has_ground, ground_equ):
    """Process each instance to generate 3D bounding boxes."""
    boxes3d = []
    center_cam_list = []
    dimension_list = []
    R_cam_list = []

    for mask_ind, cur_mask in enumerate(mask_instance):
        # Remove instances with masks that are too small which are not reliable
        if cur_mask.sum() > 0:
            min_row, min_col, max_row, max_col = find_min_max_indices(cur_mask[0])
        else:
            min_col, max_col = -1, -1
        if max_col - min_col < 3 or cur_mask.sum() < 10:
            # Fill default values for invalid masks
            boxes3d.append(np.full((8, 3), -1))
            center_cam_list.append(-1 * np.ones(3))
            dimension_list.append([-1, -1, -1])
            R_cam_list.append(-1 * np.ones((3, 3)))
            continue

        # Generate pseudo lidar data
        cur_depth = depth * cur_mask.squeeze(0)
        uv_depth = create_uv_depth(cur_depth)
        pseudo_lidar = project_image_to_cam(uv_depth, np.array(K))

        # Estimate 3D bounding box with llm-generated prior
        category_name = info_i["phrases"][mask_ind]
        prior = np.array(cat_prior[category_name])
        bbox_params = estimate_bbox(pseudo_lidar, prior, category_name, ground_equ if has_ground else None)

        boxes3d.extend(bbox_params[0])
        center_cam_list.extend(bbox_params[1])
        dimension_list.extend(bbox_params[2])
        R_cam_list.extend(bbox_params[3])

    return boxes3d, center_cam_list, dimension_list, R_cam_list


def process_outdoor(dataset, cat_prior, input_folder, output_folder):
    """Main function to process indoor data."""

    # vis_folder = os.path.join(output_folder, 'bbox_3d')
    # util.mkdir_if_missing(vis_folder)
    # vis_folder = os.path.join(output_folder, 'pseudo_lidar')
    # util.mkdir_if_missing(vis_folder)

    info = torch.load(os.path.join(input_folder, 'info.pth'))
    info_ground = torch.load(os.path.join(input_folder, 'info_ground.pth'))

    for idx in tqdm(range(len(dataset._dataset))):
        im_id = dataset._dataset[idx]['image_id']
        if im_id not in info or not info[im_id]:
            continue

        depth = np.load(f'{input_folder}/depth/{im_id}.npy')
        mask = np.load(f'{input_folder}/mask/{im_id}.npy')
        mask = adaptive_erode_mask(mask.astype(float), 4, 2, 4, 2)
        K = dataset._dataset[idx]['K']

        # Process ground data and estimate ground plane
        has_ground, ground_equ = process_ground(info_ground, im_id, depth, input_folder, K)

        # # Generate whole pseudo lidar data
        # whole_mask = mask.squeeze(1).sum(0)
        # whole_mask[whole_mask > 1] = 1
        # pseudo_lidar = create_uv_depth(depth, whole_mask)
        # pseudo_lidar = project_image_to_cam(pseudo_lidar, np.array(K))
        # np.save(f'{output_folder}/pseudo_lidar/{im_id}.npy', pseudo_lidar[:, :3])

        # Process instances and generate 3D bounding boxes
        boxes3d, center_cam_list, dimension_list, R_cam_list = process_instances(
            mask, depth, K, info[im_id], cat_prior, has_ground, ground_equ
        )

        # Update info dictionary
        info[im_id].update({
            'boxes3d': boxes3d,
            'center_cam': center_cam_list,
            'dimensions': dimension_list,
            'R_cam': R_cam_list
        })

        # # Save 3D bounding boxes
        # np.save(f'{output_folder}/bbox_3d/{im_id}.npy', np.array(boxes3d))

    # Save updated info
    torch.save(info, os.path.join(input_folder, 'info_3d.pth'))



def estimate_bbox(in_pc, prior, catgory_name, ground_equ=None):
    # Subsample input point cloud if needed
    if in_pc.shape[0] > 500:
        rand_ind = np.random.randint(0, in_pc.shape[0], 500)
        in_pc = in_pc[rand_ind]

    w, h, l = prior

    # rotate the point cloud to align with the ground plane
    if ground_equ is not None:
        dot_product = np.dot([0, -1, 0], ground_equ[:3])
        if dot_product <= 0:
            ground_equ = -ground_equ
        new_ground_equ = np.array([0, -1, 0, point_to_plane_distance(ground_equ, 0, 0, 0)])
        rotation_matrix = rotation_matrix_from_vectors([0, -1, 0], ground_equ[:3])
    else:
        rotation_matrix = np.eye(3)
    
    rotated_pc = np.dot(in_pc, rotation_matrix)

    # PCA to determine yaw
    pca = PCA(2)
    pca.fit(rotated_pc[:, [0, 2]])
    yaw_vec = pca.components_[0, :]
    yaw = np.arctan2(yaw_vec[1], yaw_vec[0])

    # Rotate the point cloud to align with the x-axis and z-axis
    rotated_pc_2 = rotate_y(yaw) @ rotated_pc.T
    x_min, x_max = rotated_pc_2[0, :].min(), rotated_pc_2[0, :].max()
    y_min, y_max = rotated_pc_2[1, :].min(), rotated_pc_2[1, :].max()
    z_min, z_max = rotated_pc_2[2, :].min(), rotated_pc_2[2, :].max()

    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2

    if not h * 0.8 <= dy <= h * 1.2 and catgory_name != 'barrier':
        dy = h
        if ground_equ is not None:
            cdis = point_to_plane_distance(new_ground_equ, cx, cy, cz)
            # Place the bottom surface of the object on the ground.
            cy += cdis - dy / 2

    vertives_list, center_cam_list, dimension_list, R_cam_list = [], [], [], []

    # If the size of the object is in a reasonable range, we will directly use it to generate the 3D bounding box.
    # Otherwise, we will try to find the more reasonable size.
    # The barrier is too flexible, so we don't limit its size.
    if (l * 0.8 <= dx <= l * 1.2 and w * 0.8 <= dz <= w * 1.2) or (l * 0.8 <= dz <= l * 1.2 and w * 0.8 <= dx <= w * 1.2) or catgory_name == 'barrier':
        vertives = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
        vertives = np.dot(rotate_y(-yaw), vertives.T).T
        vertives = np.dot(vertives, rotation_matrix.T)
        vertives_list.append(vertives)
        center_cam = vertives.mean(0)
        dimension = [dz, dy, dx]
        R_cam = rotation_matrix @ rotate_y(-yaw)
        center_cam_list.append(center_cam)
        dimension_list.append(dimension)
        R_cam_list.append(R_cam)
    else:
        # generate all the proposal boxes.
        possible_bboxs = generate_possible_bboxs(cx, cz, dx, dz, w, l)
        min_loss, min_vertives = float('inf'), None
        
        # find the best proposal box.
        for possible_bbox in possible_bboxs:
            x_min, x_max, z_min, z_max = possible_bbox
            dx, dz = x_max - x_min, z_max - z_min
            cx, cz = (x_min + x_max) / 2, (z_min + z_max) / 2
            inside_ratio = calc_inside_ratio(rotated_pc_2, x_min, x_max, z_min, z_max)
            vertives = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
            vertives = np.dot(rotate_y(-yaw), vertives.T).T
            new_cx, new_cz = vertives[:, 0].mean(), vertives[:, 2].mean()

            # calculate the ray tracing loss and inside ratio loss.
            pc_tensor = torch.from_numpy(rotated_pc).float()
            loss_ray_tracing = calc_dis_ray_tracing(torch.Tensor([dz, dx]), torch.Tensor([yaw]), pc_tensor[:, [0, 2]], torch.Tensor([new_cx, new_cz]))
            loss_inside_ratio = 1 - inside_ratio

            loss = loss_ray_tracing + 10 * loss_inside_ratio

            if loss < min_loss:
                min_loss = loss
                min_vertives = vertives
        
        min_vertives = np.dot(min_vertives, rotation_matrix.T)
        vertives_list.append(min_vertives)
        center_cam = min_vertives.mean(0)
        dimension = [dz, dy, dx]
        R_cam = rotation_matrix @ rotate_y(-yaw)
        center_cam_list.append(center_cam)
        dimension_list.append(dimension)
        R_cam_list.append(R_cam)

    return vertives_list, center_cam_list, dimension_list, R_cam_list