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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_colormap(cmap_name):
    """
    Draw a color bar.
    @param cmap_name: Name of the colormap.
    @return: Displays the color bar.
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
    Convert numpy array to point cloud.
    @param points: Numpy array of point cloud data.
    @return: Open3D PointCloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def get_color_map(x, cmap_name="Oranges"):
    """
    Get colors from a colormap.
    :param x: Input values to map colors.
    :param cmap_name: Colormap name, e.g. "Oranges", "jet".
    :return: Array of RGB colors.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Retrieve the colormap 'cmap_name' with 8 discrete colors.
    # ------------------------------------------------------------------------------------------------------------------
    viridis = cm.get_cmap(cmap_name, 8)
    colours = viridis(x).squeeze()
    # ------------------------------------------------------------------------------------------------------------------
    # Optionally plot the colormap bar.
    # ------------------------------------------------------------------------------------------------------------------
    # plot_colormap(cmap_name)
    return colours[:, :3]


def vis_points(points_xyz, file_name):
    """
    Visualize point cloud data.
    @param points_xyz: Point cloud data as ndarray.
    @param file_name: Filename (string) for saving the point cloud.
    @return: None
    """
    # ------------------------------------------------------------------------------------------------------------------
    # These two lines are necessary.
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
    # ------------------------------------------------------------------------------------------------------------------
    # Set uniform color for the initial point cloud display.
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    # ------------------------------------------------------------------------------------------------------------------
    # Save and visualize the point cloud.
    # ------------------------------------------------------------------------------------------------------------------
    filename = f"output/{file_name}.ply"
    o3d.io.write_point_cloud(filename=filename, pointcloud=point_cloud)
    o3d.visualization.draw_geometries([point_cloud], window_name="dam visualization")


def vis_points_knn(points_xyz, center_index, knn_index, file_name=None):
    """
    Visualize sparse points and their neighborhoods on a dense point cloud.
    @param center_index: Index of the sparse point on the dense point cloud, ndarray, ()
    @param knn_index: Neighborhood indices of the sparse point on the dense point cloud, ndarray, (16,)
    @param points_xyz: Coordinates of the dense point cloud (xyz), ndarray, (1024, 3)
    @param file_name: Filename to save the point cloud
    @return: None
    """
    # ------------------------------------------------------------------------------------------------------------------
    # These two lines are necessary
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
    # ------------------------------------------------------------------------------------------------------------------
    # Set initial color for the point cloud
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    # ------------------------------------------------------------------------------------------------------------------
    # Color neighborhood points green
    # ------------------------------------------------------------------------------------------------------------------
    np.asarray(point_cloud.colors)[knn_index] = [0.0, 1.0, 0.0]
    # ------------------------------------------------------------------------------------------------------------------
    # Color the query (center) point red
    # ------------------------------------------------------------------------------------------------------------------
    np.asarray(point_cloud.colors)[center_index] = [1.0, 0.0, 0.0]
    # ------------------------------------------------------------------------------------------------------------------
    # Visualize and save
    # ------------------------------------------------------------------------------------------------------------------
    filename = f"output/{file_name}.ply"
    o3d.io.write_point_cloud(filename=filename, pointcloud=point_cloud)
    o3d.visualization.draw_geometries([point_cloud], window_name="dam visualization")


def vis_points_with_label(points_xyz, points_seg_label, file_name=None):
    """
    Visualize point cloud segmentation results.
    @param points_xyz: Point cloud data, first 3 dims XYZ, ndarray
    @param points_seg_label: Segmentation labels of points, ndarray
    @param file_name: Filename to save the point cloud, str
    @return: None
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Define colors for parts segmentation, values in [0, 1]
    # ------------------------------------------------------------------------------------------------------------------
    colors = []
    for r in [0.0, 0.33, 0.66, 1.0]:
        for g in [0.0, 0.33, 0.66, 1.0]:
            for b in [0.0, 0.33, 0.66, 1.0]:
                colors.append([r, g, b])
    # ------------------------------------------------------------------------------------------------------------------
    # These two lines are necessary
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
    # ------------------------------------------------------------------------------------------------------------------
    # Set initial color of the point cloud to black
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud.paint_uniform_color([0, 0, 0])
    if points_seg_label is not None:
        # --------------------------------------------------------------------------------------------------------------
        # Assign different colors based on segmentation labels
        # --------------------------------------------------------------------------------------------------------------
        min_label = np.min(points_seg_label)
        for i in range(len(points_seg_label)):
            point_cloud.colors[i] = colors[int(points_seg_label[i] - min_label)]
    # ------------------------------------------------------------------------------------------------------------------
    # Visualize and save
    # ------------------------------------------------------------------------------------------------------------------
    filename = f"output/{file_name}.ply"
    o3d.io.write_point_cloud(filename=filename, pointcloud=point_cloud)
    o3d.visualization.draw_geometries([point_cloud], window_name="dam visualization")


def view_points_with_attn_score(point_np, scores, file_name):
    """
    Visualize attention scores on points.
    @param point_np: ndarray, shape = (2048, 3)
    @param scores: ndarray, shape = (2048,)
    @param file_name: string
    @return: None
    """
    point_pcd = get_pcd(point_np)
    # ------------------------------------------------------------------------------------------------------------------
    # Set all points to grey
    # ------------------------------------------------------------------------------------------------------------------
    point_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # ------------------------------------------------------------------------------------------------------------------
    # Color each point by its attention score mapped to color
    # ------------------------------------------------------------------------------------------------------------------
    colors = get_color_map(scores)
    for index, weight in enumerate(scores):
        np.asarray(point_pcd.colors)[index, :] = colors[index]
    # ------------------------------------------------------------------------------------------------------------------
    # Save and visualize
    # ------------------------------------------------------------------------------------------------------------------
    o3d.io.write_point_cloud(filename=f"output/vis/{file_name}.ply", pointcloud=point_pcd)
    o3d.visualization.draw_geometries([point_pcd], window_name="dam visualization")


def normalize_pytorch(data):
    """
    Normalize data along the last dimension.
    @param data: (B, N, C).
    @return: normalized data in the same shape
    """
    data_min = torch.min(data, dim=-1, keepdim=True)[0]
    data_max = torch.max(data, dim=-1, keepdim=True)[0]
    result = (data - data_min) / (data_max - data_min + 1e-8)
    return result


def vis_matrix(cost):
    """
    Visualize a normalized matrix as a heatmap.
    @param cost: input matrix (tensor)
    @return: None
    """
    normalized_matrix = normalize_pytorch(cost)
    normalized_matrix_np = normalized_matrix.detach().cpu().numpy()[0]
    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_matrix_np, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Normalized Matrix Heatmap')
    plt.xlabel('Feature Index')
    plt.ylabel('Sample Index')
    plt.show()


def visualize_distribution(student_outputs, teacher_outputs, epoch, title="Scatter Visualization"):
    """
    Scatter plot visualization of student vs teacher output feature distributions.
    @param student_outputs: list/array of student model output features
    @param teacher_outputs: list/array of teacher model output features
    @param epoch: current training epoch
    @param title: plot title
    @return: None
    """
    student_outputs = np.concatenate(student_outputs, axis=0)
    teacher_outputs = np.concatenate(teacher_outputs, axis=0)
    sample_indices = np.random.choice(student_outputs.shape[0], size=1000, replace=False)
    student_select = student_outputs[sample_indices]
    teacher_select = teacher_outputs[sample_indices]
    # Scatter plot: student output vs teacher output
    plt.figure(figsize=(20, 8))
    plt.scatter(student_select.flatten(), teacher_select.flatten(), alpha=0.5)
    plt.xlabel("Student Output")
    plt.ylabel("Teacher Output")
    plt.title(title)
    plt.savefig(f"output/fea_dis/distribution_{epoch}.png")
    plt.close()


def visualize_tsne(features, labels, epoch, title="t-SNE Visualization", num_classes=40):
    """
    Perform t-SNE dimensionality reduction and visualize features colored by class labels.
    @param features: torch.Tensor feature matrix
    @param labels: torch.Tensor label vector
    @param epoch: current training epoch
    @param title: plot title
    @param num_classes: number of classes to visualize
    @return: None
    """
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()

    # Perform TSNE dimensionality reduction to 2D
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    classnames = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
        'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
        'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
        'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
        'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]

    # Generate all RGB combinations (64 total)
    rgb_values = [0.0, 0.33, 0.66, 1.0]
    all_colors = np.array([[r, g, b] for r in rgb_values for g in rgb_values for b in rgb_values])

    # Sample 40 colors uniformly for classes
    indices = np.linspace(0, len(all_colors) - 1, num_classes, dtype=int)
    colors = all_colors[indices]

    plt.figure(figsize=(12, 8))

    # Scatter plot colored by class
    for i in range(num_classes):
        plt.scatter(features_2d[labels == i, 0], features_2d[labels == i, 1],
                    color=colors[i], label=classnames[i], s=40, alpha=0.6)

    # Add legend outside plot
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, ncol=1, frameon=False)
    plt.title(title)
    plt.savefig(f"output/fea_dis/tsne_{epoch}.png", bbox_inches='tight')
    plt.close()


def visualize_pca_and_save(ori_pc, encoder_features, center_points, index_num):
    """
    Apply PCA to encoder features, map to RGB colors, and save center points as a colored PLY file.
    @param ori_pc: original point cloud (tensor)
    @param encoder_features: features from encoder (tensor)
    @param center_points: center points to save (tensor)
    @param index_num: index number used for file naming
    @return: None
    """
    ori_pc = ori_pc.detach().cpu().numpy()
    features = encoder_features.detach().cpu().numpy()
    center_points = center_points.detach().cpu().numpy()

    # PCA reduction to 3 dimensions
    pca = PCA(n_components=3)
    color_map = pca.fit_transform(features)

    # Normalize colors to [0, 1]
    color_map -= color_map.min(axis=0)
    color_map /= color_map.max(axis=0)

    # Save center points as PLY with PCA colors
    center_pcd = o3d.geometry.PointCloud()
    center_pcd.points = o3d.utility.Vector3dVector(center_points)
    center_pcd.colors = o3d.utility.Vector3dVector(color_map)
    o3d.io.write_point_cloud(f"output/vis/pca_{index_num}.ply", center_pcd)
    print(f"Center points saved to output/vis/center_{index_num}.ply")
