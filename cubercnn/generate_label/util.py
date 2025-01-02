# Copyright (c) Meta Platforms, Inc. and affiliates
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from matplotlib.path import Path
from cubercnn import util
from sklearn.decomposition import PCA
import math


def extract_ground(points):
    a1,b1,c1,d1,distance_1 = fit_plane_x(points)
    a2,b2,c2,d2,distance_2 = fit_plane_y(points)
    a3,b3,c3,d3,distance_3 = fit_plane_z(points)

    if distance_1 < distance_2 and distance_1 < distance_3:
        return np.array([a1, b1, c1, d1])
    elif distance_2 < distance_1 and distance_2 < distance_3:
        return np.array([a2, b2, c2, d2])
    else:
        return np.array([a3, b3, c3, d3])


def fit_plane_x(points):
    X = np.c_[points[:, [0,1]], np.ones(points.shape[0])]
    Y = points[:, 2]
    try:
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    except(np.linalg.LinAlgError):
        return np.inf, np.inf, np.inf, np.inf, np.inf
    a, b, d = w
    c = -1
    Y_pred = X.dot(w)
    mse = np.mean((Y - Y_pred) ** 2)
    return a, b, c, d, mse

def fit_plane_y(points):
    X = np.c_[points[:, [0,2]], np.ones(points.shape[0])]
    Y = points[:, 1]
    try:
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    except(np.linalg.LinAlgError):
        return np.inf, np.inf, np.inf, np.inf, np.inf
    a, c, d = w
    b = -1
    Y_pred = X.dot(w)
    mse = np.mean((Y - Y_pred) ** 2)
    return a, b, c, d, mse

def fit_plane_z(points):
    X = np.c_[points[:, [1,2]], np.ones(points.shape[0])]
    Y = points[:, 0]
    try:
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    except(np.linalg.LinAlgError):
        return np.inf, np.inf, np.inf, np.inf, np.inf
    b, c, d = w
    a = -1
    Y_pred = X.dot(w)
    mse = np.mean((Y - Y_pred) ** 2)
    return a, b, c, d, mse



def project_image_to_cam(uv_depth, K):
    ''' Input: nx3 first two channels are uv, 3rd channel
                is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    c_u = K[0,2]
    c_v = K[1,2]
    f_u = K[0,0]
    f_v = K[1,1]

    n = uv_depth.shape[0]
    x = ((uv_depth[:,0]-c_u)*uv_depth[:,2])/f_u
    y = ((uv_depth[:,1]-c_v)*uv_depth[:,2])/f_v
    pts_3d_rect = np.zeros((n,3))
    pts_3d_rect[:,0] = x
    pts_3d_rect[:,1] = y
    pts_3d_rect[:,2] = uv_depth[:,2]
    return pts_3d_rect



def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Function to obtain the rotation matrix that rotates vec1 to vec2.
    """
    vec1 = normalize(vec1)
    vec2 = normalize(vec2)
    
    axis = np.cross(vec1, vec2)
    cos_theta = np.dot(vec1, vec2)
    
    skew_symmetric = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + skew_symmetric + np.dot(skew_symmetric, skew_symmetric) * (1 - cos_theta) / (np.linalg.norm(axis) ** 2)
    
    return rotation_matrix

def rotate_y(angle):
    """
    Function to obtain the rotation matrix around y-axis according to the given angle.
    """
    rotmat = np.zeros((3, 3))
    rotmat[1, 1] = 1
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotmat[0, 0] = cosval
    rotmat[0, 2] = sinval
    rotmat[2, 0] = -sinval
    rotmat[2, 2] = cosval
    return rotmat



def point_to_plane_distance(ground_equ, x, y, z):
    A, B, C, D = ground_equ
    plane_normal_length = np.sqrt(A**2 + B**2 + C**2)
    distance = abs(A*x + B*y + C*z + D) / plane_normal_length
    
    return distance



def convert_box_vertices(center_x, center_y, center_z, l, w, h, yaw):
    # Define the corners of the box in its local coordinate system
    local_corners = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2]
    ])

    # Rotation matrix around y-axis (yaw)
    rotation_matrix = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])

    # Apply rotation to local corners
    rotated_corners = np.dot(local_corners, rotation_matrix.T)

    # Translate to global coordinates
    global_corners = rotated_corners + np.array([center_x, center_y, center_z])

    return global_corners



def scale_rectangle(rectangle, length_left, length_right, origin_vertex):
    scaled_rectangles = [None] * 4
    scaled_rectangles[origin_vertex] = rectangle[origin_vertex]
    origin_x, origin_y = rectangle[origin_vertex]

    # left 
    x, y = rectangle[(origin_vertex-1)%4]
    relative_x = x - origin_x
    relative_y = y - origin_y
    scale_left = length_left / np.linalg.norm([relative_x, relative_y])
    x_scaled_left = relative_x * scale_left
    y_scaled_left = relative_y * scale_left
    x_scaled = origin_x + x_scaled_left
    y_scaled = origin_y + y_scaled_left
    scaled_rectangles[(origin_vertex-1)%4] = (x_scaled, y_scaled)

    # right
    x, y = rectangle[(origin_vertex+1)%4]
    relative_x = x - origin_x
    relative_y = y - origin_y
    scale_right = length_right / np.linalg.norm([relative_x, relative_y])
    x_scaled_right = relative_x * scale_right
    y_scaled_right = relative_y * scale_right
    x_scaled = origin_x + x_scaled_right
    y_scaled = origin_y + y_scaled_right
    scaled_rectangles[(origin_vertex+1)%4] = (x_scaled, y_scaled)

    # opposite
    x, y = rectangle[(origin_vertex+2)%4]
    x_scaled = origin_x + x_scaled_right + x_scaled_left
    y_scaled = origin_y + y_scaled_right + y_scaled_left
    scaled_rectangles[(origin_vertex+2)%4] = (x_scaled, y_scaled)
    return scaled_rectangles


def generate_scaled_rectangles(rectangle, w, l):
    """
    Generate all possible scaled rectangles based on the corners of the original rectangle, along with the desired width and length.
    """
    scaled_rectangles = []
    
    # For each vertex of the rectangle, generate scaled rectangles
    for i in range(4):
        scaled_rectangle1 = scale_rectangle(rectangle, w, l, i)
        scaled_rectangle2 = scale_rectangle(rectangle, l, w, i)
        scaled_rectangles.extend([scaled_rectangle1, scaled_rectangle2])
    
    return scaled_rectangles


def generate_possible_bboxs(cx, cz, dx, dz, w, l):
    """
    Generate all possible bounding boxes given the center and dimensions of the box, as well as the desired width and length.
    """
    init_theta, length = np.arctan(dz / dx), np.sqrt(dx ** 2 + dz ** 2) / 2

    def calculate_corner(theta):
        return (length * np.cos(theta) + cx, length * np.sin(theta) + cz)
    
    corners = [
        calculate_corner(init_theta),
        calculate_corner(np.pi - init_theta),
        calculate_corner(np.pi + init_theta),
        calculate_corner(-init_theta),
    ]

    scaled_rectangles = generate_scaled_rectangles(corners, w, l)
    transform_scaled_rectangles = []
    for i in range(len(scaled_rectangles)):
        corners = np.array(scaled_rectangles[i])
        x_min = corners[:,0].min()
        x_max = corners[:,0].max()
        z_min = corners[:,1].min()
        z_max = corners[:,1].max()
        transform_scaled_rectangles.append([x_min, x_max, z_min, z_max])
    return transform_scaled_rectangles



def find_min_max_indices(mask):
    indices = np.argwhere(mask == 1)
    min_row, min_col = np.min(indices, axis=0)
    max_row, max_col = np.max(indices, axis=0)
    return min_row, min_col, max_row, max_col


def erode_mask(mask, k_vertical, k_horizontal):
    """
    Function to erode the mask using vertical and horizontal kernels.
    """
    new_mask = np.zeros_like(mask)
    kernel_vertical = np.ones((3,1), np.uint8)  
    kernel_horizontal = np.ones((1,3), np.uint8)  
    
    for i in range(mask.shape[0]):
        mask_i = mask[i][0]
        eroded_mask_vertical = cv2.erode(mask_i, kernel_vertical, iterations=k_vertical)
        eroded_mask_horizontal = cv2.erode(mask_i, kernel_horizontal, iterations=k_horizontal)
        new_mask[i][0] = np.logical_and(eroded_mask_vertical, eroded_mask_horizontal).astype(np.uint8)
    return new_mask


def adaptive_erode_mask(mask, k_vertical, k_vertical_min, k_horizontal, k_horizontal_min):
    """
    Function to erode the mask based on the size of the mask.
    If the mask is too small, use the minimum kernel size.
    """ 
    new_mask = np.zeros_like(mask)
    kernel_vertical = np.ones((3,1), np.uint8)  
    kernel_horizontal = np.ones((1,3), np.uint8)  
    
    for i in range(mask.shape[0]):
        mask_i = mask[i][0]
        min_row, min_col, max_row, max_col = find_min_max_indices(mask_i)

        k_vertical = k_vertical if max_row - min_row >= 10 else k_vertical_min
        k_horizontal = k_horizontal if max_col - min_col >= 10 else k_horizontal_min
        
        eroded_mask_vertical = cv2.erode(mask_i, kernel_vertical, iterations=k_vertical)
        eroded_mask_horizontal = cv2.erode(mask_i, kernel_horizontal, iterations=k_horizontal)

        new_mask[i][0] = np.logical_and(eroded_mask_vertical, eroded_mask_horizontal).astype(np.uint8)
    return new_mask
