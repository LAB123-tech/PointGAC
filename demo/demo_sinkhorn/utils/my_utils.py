import numpy as np
import open3d as o3d
import torch


def farthest_point_sample_numpy(point, n_point):
    """
    Farthest point sampling based on numpy
    @param point: (2601, 3)
    @param n_point: int, 2048
    @return: index, 2048
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((n_point,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_point):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    index = centroids.astype(np.int32)
    point = point[index, :]
    return point


def square_distance(src, dst):
    """
    Compute Euclidean distance between points using torch
    @param src: [B, N, C]
    @param dst: [B, M, C]
    @return: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def pc_normalize(pc):
    """
    Normalize point cloud
    @param pc:
    @return:
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def estimate_normal(pcd, radius=0.06, max_nn=30):
    """
    Estimate normals of the point cloud
    @param pcd:
    @param radius:
    @param max_nn:
    @return:
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
