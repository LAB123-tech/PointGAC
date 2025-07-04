# -*- coding: utf-8 -*-
# @Time    : 2024-04-16
# @Author  : lab
# @desc    :

import matplotlib
import numpy as np
import open3d as o3d
import torch
from matplotlib import cm
from matplotlib import pyplot as plt


def plot_colormap(cmap_name):
    """
    Draw a progress bar.
    @param cmap_name: The name of the progress bar.
    @return: bar
    """
    fig, ax = plt.subplots()
    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, cmap.N))
    ax.imshow([colors], extent=[0, 10, 0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(cmap_name)
    # plt.savefig("a.png", bbox_inches='tight')
    plt.show()


def get_pcd(points):
    """
    Data type conversion, numpy -> point cloud
    @param points: 点云numpy数据
    @return: 点云point cloud类型数据
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def get_color_map(x, cmap_name="Oranges"):
    """
    Set color type
    :param x:
    :param cmap_name:
    :return:
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Get the colormap named cmap_name='Oranges', can also set to 'jet'
    # ------------------------------------------------------------------------------------------------------------------
    viridis = cm.get_cmap(cmap_name, 8)
    colours = viridis(x).squeeze()
    # ------------------------------------------------------------------------------------------------------------------
    # Draw a progress bar
    # ------------------------------------------------------------------------------------------------------------------
    # plot_colormap(cmap_name)
    return colours[:, :3]


def vis_points(points_xyz, file_name):
    """
    Visualize point cloud object
    @param points_xyz: point cloud data, ndarray
    @param file_name: point cloud name -> str
    @return: None
    """
    # ------------------------------------------------------------------------------------------------------------------
    # These two lines are necessary
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
    # ------------------------------------------------------------------------------------------------------------------
    # Set the initial display color of the point cloud
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    # ------------------------------------------------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------------------------------------------------
    filename = "output/{}.ply".format(file_name)
    o3d.io.write_point_cloud(filename=filename, pointcloud=point_cloud)
    o3d.visualization.draw_geometries([point_cloud], window_name="dam visualization")


def vis_points_knn(points_xyz, center_index, knn_index, file_name=None):
    """
    Visualize sparse points and their neighborhood on dense point cloud
    @param center_index: indices of sparse points on dense point cloud, ndarray, ()
    @param knn_index: neighborhood indices of sparse points on dense point cloud, ndarray, (16,)
    @param points_xyz: dense point cloud coordinates xyz, ndarray, (1024, 3)
    @param file_name: saved point cloud name
    @return: None
    """
    # ------------------------------------------------------------------------------------------------------------------
    # These two lines are necessary
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
    # ------------------------------------------------------------------------------------------------------------------
    # Set the initial display color of the point cloud
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    # ------------------------------------------------------------------------------------------------------------------
    # Color neighborhood points green
    # ------------------------------------------------------------------------------------------------------------------
    np.asarray(point_cloud.colors)[knn_index] = [0.0, 1.0, 0.0]
    # ------------------------------------------------------------------------------------------------------------------
    # Render query point in red
    # ------------------------------------------------------------------------------------------------------------------
    np.asarray(point_cloud.colors)[center_index] = [1.0, 0.0, 0.0]
    # ------------------------------------------------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------------------------------------------------
    filename = "output/{}.ply".format(file_name)
    o3d.io.write_point_cloud(filename=filename, pointcloud=point_cloud)
    o3d.visualization.draw_geometries([point_cloud], window_name="dam visualization")


def vis_points_with_label(points_xyz, points_seg_label, file_name=None):
    """
    Visualize point cloud segmentation results
    @param points_xyz: point cloud data, first three dims XYZ -> ndarray
    @param points_seg_label: point cloud segmentation labels -> ndarray
    @param file_name: point cloud name -> str
    @return: None
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Define 64 part colors, values range in [0, 1]
    # ------------------------------------------------------------------------------------------------------------------
    colors = []
    steps = np.linspace(0.0, 1, 3)  # 3 levels: 0.0 ~ 1.0, total 27 combinations
    for r in steps:
        for g in steps:
            for b in steps:
                colors.append([r, g, b])
    # ------------------------------------------------------------------------------------------------------------------
    # These two lines are necessary
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
    # ------------------------------------------------------------------------------------------------------------------
    # Set the initial display color of the point cloud
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud.paint_uniform_color([0, 0, 0])
    if points_seg_label is not None:
        # --------------------------------------------------------------------------------------------------------------
        # Set parts to different colors based on different label values
        # --------------------------------------------------------------------------------------------------------------
        min_label = np.min(points_seg_label)
        for i in range(len(points_seg_label)):
            # point_cloud.colors[i] = colors[int(points_seg_label[i] - min_label)]
            point_cloud.colors[i] = colors[int(points_seg_label[i]) + 10]
    # ------------------------------------------------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------------------------------------------------
    filename = "output/{}.ply".format(file_name)
    o3d.io.write_point_cloud(filename=filename, pointcloud=point_cloud)
    o3d.visualization.draw_geometries([point_cloud], window_name="dam visualization")


def vis_points_with_label_512(points_xyz, points_seg_label, file_name=None):
    """
    Visualize point cloud segmentation results
    @param points_xyz: point cloud data, first three dims XYZ -> ndarray
    @param points_seg_label: point cloud segmentation labels -> ndarray
    @param file_name: point cloud name -> str
    @return: None
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Define 64 part colors, values range in [0, 1] with 4*4*4 combinations
    # ------------------------------------------------------------------------------------------------------------------
    colors = []
    steps = np.linspace(0.0, 1, 4)
    for r in steps:
        for g in steps:
            for b in steps:
                colors.append([r, g, b])
    # ------------------------------------------------------------------------------------------------------------------
    # These two lines are necessary
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
    # ------------------------------------------------------------------------------------------------------------------
    # Set the initial display color of the point cloud
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud.paint_uniform_color([0, 0, 0])
    if points_seg_label is not None:
        # --------------------------------------------------------------------------------------------------------------
        # Set parts to different colors based on different label values
        # --------------------------------------------------------------------------------------------------------------
        min_label = np.min(points_seg_label)
        for i in range(len(points_seg_label)):
            point_cloud.colors[i] = colors[int(points_seg_label[i] - min_label)]
    # ------------------------------------------------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------------------------------------------------
    filename = "output/{}.ply".format(file_name)
    o3d.io.write_point_cloud(filename=filename, pointcloud=point_cloud)
    o3d.visualization.draw_geometries([point_cloud], window_name="dam visualization")


def view_points_with_attn_score(point_np, scores, center_index, file_name):
    """
    Visualize attention scores
    @param point_np: ndarray, shape = (2048, 3)
    @param scores: ndarray, shape = (2048,)
    @param center_index: int
    @param file_name: string
    @return: ndarray, shape = (2048, 3)
    """
    point_pcd = get_pcd(point_np)
    # ------------------------------------------------------------------------------------------------------------------
    # Change all point cloud data color to gray
    # ------------------------------------------------------------------------------------------------------------------
    point_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # ------------------------------------------------------------------------------------------------------------------
    # Color each point according to its corresponding score color
    # ------------------------------------------------------------------------------------------------------------------
    colors = get_color_map(scores)
    for index, weight in enumerate(scores):
        np.asarray(point_pcd.colors)[index, :] = colors[index]
    # ------------------------------------------------------------------------------------------------------------------
    # Render query point in green
    # ------------------------------------------------------------------------------------------------------------------
    np.asarray(point_pcd.colors)[center_index] = [0.0, 1.0, 0.0]
    o3d.io.write_point_cloud(filename="output/{}.ply".format(file_name), pointcloud=point_pcd)
    o3d.visualization.draw_geometries([point_pcd], window_name="dam visualization")


def normalize_pytorch(data):
    """
    Normalize data along the last dimension
    @param data: (B, N, C).
    @return:
    """
    data_min = torch.min(data, dim=-1, keepdim=True)[0]
    data_max = torch.max(data, dim=-1, keepdim=True)[0]
    result = (data - data_min) / (data_max - data_min + 1e-8)
    return result


def vis_matrix(cost):
    normalized_matrix = normalize_pytorch(cost)
    normalized_matrix_np = normalized_matrix.detach().cpu().numpy()[0]
    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_matrix_np, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Normalized Matrix Heatmap')
    plt.xlabel('Feature Index')
    plt.ylabel('Sample Index')
    plt.show()
