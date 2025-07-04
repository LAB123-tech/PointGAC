# -*- coding: utf-8 -*-
# @Time    : 2024-10-29
# @Author  : lab
# @desc    :
import sys

import numpy as np

from utils.my_utils import *
from utils.sinkhorn import sinkhorn_knopp
from utils.visualize import vis_points_with_label, vis_points_with_label_512

sys.path.append("/home/lab/文档/PointGAC/lib/Partition_lib/cut-pursuit/build/src")
sys.path.append("/home/lab/文档/PointGAC/lib/Partition_lib/ply_c")
import libply_c
import libcp
from datasets.dataset_utils.DatasetFunc import *

# ----------------------------------------------------------------------------------------------------------------------
# 获取点的位置 (N, 3) 并转换为 float32 类型
# ----------------------------------------------------------------------------------------------------------------------
pcd = o3d.io.read_point_cloud("data/phone.ply")
point_data = np.asarray(pcd.points, dtype=np.float32)
vis_points_with_label(point_data, None, "ori")
# ----------------------------------------------------------------------------------------------------------------------
# 对点云进行同质的几何切分，得到在几何上的分类结果，0.2越大，切块就越大
# ----------------------------------------------------------------------------------------------------------------------
graph_nn, target_fea = compute_graph_nn_2(point_data, 10, 30)
graph_nn["edge_weight"] = 1. / (1.0 + graph_nn["distances"] / np.mean(graph_nn["distances"]))
point_cloud_geo = libply_c.compute_geof(point_data, target_fea, 30).astype('float32')
point_cloud_geo[:, 1] = 2 * point_cloud_geo[:, 1]
components, point_partition = libcp.cutpursuit(point_cloud_geo, graph_nn["source"], graph_nn["target"],
                                               graph_nn["edge_weight"], 0.2)
vis_points_with_label(point_data, point_partition, "graph_cut_ori")
point_partition = np.array(point_partition).astype('int64').reshape(-1, 1)
point_data = np.concatenate((point_data, point_partition), axis=-1)
# ----------------------------------------------------------------------------------------------------------------------
# FPS采样
# ----------------------------------------------------------------------------------------------------------------------
center_num = 64
point_sample = farthest_point_sample_numpy(point_data, center_num)
vis_points_with_label(point_sample[:, :3], point_sample[:, -1], "far")
# ----------------------------------------------------------------------------------------------------------------------
# 计算代价矩阵
# ----------------------------------------------------------------------------------------------------------------------
point_data = torch.Tensor(point_data[None, :, :])
point_data_partition = point_data[0, :, 3].long()
point_sample = torch.Tensor(point_sample[None, :, :])
point_sample_partition = point_sample[0, :, 3].long()
cost_xyz = square_distance(point_data[:, :, :3], point_sample[:, :, :3])
# ----------------------------------------------------------------------------------------------------------------------
# 生成掩码
# ----------------------------------------------------------------------------------------------------------------------
expanded_labels_8192 = point_data_partition.reshape(-1, 1).repeat(1, center_num)  # (8192, 64)
center_point_labels_128 = point_sample_partition.reshape(1, -1).repeat(point_data.shape[1], 1).long()  # (8192, 64)
mask = (expanded_labels_8192 == center_point_labels_128).int()  # (8192, 64)
# ----------------------------------------------------------------------------------------------------------------------
# 基于最优传输进行切块
# ----------------------------------------------------------------------------------------------------------------------
n_points, n_centers = cost_xyz.shape[1], cost_xyz.shape[2]
num_lable_1 = torch.bincount(point_data_partition.flatten())  # 得到每个类别的点数量
num_class_ori = num_lable_1[point_data_partition.flatten()].reshape(-1, 1)  # 每个点类别标签对应的点数量
num_lable_2 = torch.bincount(point_sample_partition.flatten())  # 得到每个类别的点数量
num_class_center = num_lable_2[point_sample_partition.flatten()].reshape(-1, 1)  # 每个点类别标签对应的点数量
r = torch.ones(1, n_points) / num_class_ori.transpose(0, 1)  # 原始点云的分布 (1, 8192)
c = torch.ones(1, n_centers) / num_class_center.transpose(0, 1)  # 中心点的分布 (1, 64)
T, u, v = sinkhorn_knopp(r, c, cost_xyz, mask=mask)  # 使用Sinkhorn 进行均等切分
assignments = T.argmax(dim=2).squeeze(0)  # # 每个点云分区的索引 (N,)
vis_points_with_label_512(point_data[0, :, :3].cpu().numpy(), assignments.cpu().numpy(), "OT")
