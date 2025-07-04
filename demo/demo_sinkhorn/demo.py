# -*- coding: utf-8 -*-
# @Time    : 2024-10-29
# @Author  : lab
# @desc    :

from utils.my_utils import *
from utils.sinkhorn import sinkhorn_knopp
from utils.visualize import vis_points_with_label, vis_points_with_label_512

point_data = np.load('data/pointcloud.npy')
vis_points_with_label(point_data[:, :3], point_data[:, -1], "geo")
# ----------------------------------------------------------------------------------------------------------------------
# FPS
# ----------------------------------------------------------------------------------------------------------------------
point_sample = farthest_point_sample_numpy(point_data, 64)
vis_points_with_label(point_sample[:, :3], point_sample[:, -1], "far")
# ----------------------------------------------------------------------------------------------------------------------
# Compute cost matrix
# ----------------------------------------------------------------------------------------------------------------------
point_data = torch.Tensor(point_data[None, :, :])
point_data_partition = point_data[0, :, 3].long()
point_sample = torch.Tensor(point_sample[None, :, :])
point_sample_partition = point_sample[0, :, 3].long()
cost_xyz = square_distance(point_data[:, :, :3], point_sample[:, :, :3])
# ----------------------------------------------------------------------------------------------------------------------
# Generate mask
# ----------------------------------------------------------------------------------------------------------------------
expanded_labels_8192 = point_data_partition.reshape(-1, 1).repeat(1, 64)  # (8192, 64)
center_point_labels_128 = point_sample_partition.reshape(1, -1).repeat(point_data.shape[1], 1).long()  # (8192, 64)
mask = (expanded_labels_8192 == center_point_labels_128).int()  # (8192, 64)
# ----------------------------------------------------------------------------------------------------------------------
# Partitioning blocks based on optimal transport
# ----------------------------------------------------------------------------------------------------------------------
n_points, n_centers = cost_xyz.shape[1], cost_xyz.shape[2]
num_lable_1 = torch.bincount(point_data_partition.flatten())  # Get the number of points for each category
num_class_ori = num_lable_1[point_data_partition.flatten()].reshape(-1, 1)  # The number of points corresponding to each point category label
num_lable_2 = torch.bincount(point_sample_partition.flatten())  # Get the number of points for each category
num_class_center = num_lable_2[point_sample_partition.flatten()].reshape(-1, 1)  # The number of points corresponding to each point category label
r = torch.ones(1, n_points) / num_class_ori.transpose(0, 1)  # Original point cloud (1, 8192)
c = torch.ones(1, n_centers) / num_class_center.transpose(0, 1)  # center points (1, 64)
T, u, v = sinkhorn_knopp(r, c, cost_xyz, mask=mask)  # Use Sinkhorn for balanced partitioning
assignments = T.argmax(dim=2).squeeze(0)  # Indices of each point cloud partition (N,)
vis_points_with_label_512(point_data[0, :, :3].cpu().numpy(), assignments.cpu().numpy(), "OT")
