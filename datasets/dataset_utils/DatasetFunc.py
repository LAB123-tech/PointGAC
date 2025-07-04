import os

import h5py
import numpy as np
import open3d
import torch
from sklearn.neighbors import NearestNeighbors


def farthest_point_sample_numpy(point, n_point):
    """
    Farthest point sampling based on NumPy.
    @param point: ndarray(2601, 3)
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


def pc_norm_torch(pc):
    """
    Normalize the data based on torch
    @param pc: (B, N, D)，D >= 3
    @return: (B, N, D)，
    """
    centroid = torch.mean(pc[:, :, :3], dim=1, keepdim=True)
    pc[:, :, :3] = pc[:, :, :3] - centroid
    m = torch.max(torch.sqrt(torch.sum(pc[:, :, :3] ** 2, dim=2, keepdim=True)), dim=1, keepdim=True)[0]
    pc[:, :, :3] = pc[:, :, :3] / m
    return pc


def pc_norm_numpy(pc):
    """
    Normalize the data based on NumPy.
    @param pc: (N, 3)
    @return: (N, 3)
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class IO:
    @classmethod
    def get(cls, file_path):
        try:
            _, file_extension = os.path.splitext(file_path)
            if file_extension in ['.npy']:
                return cls._read_npy(file_path)
            elif file_extension in ['.h5']:
                return cls._read_h5(file_path)
            elif file_extension in ['.txt']:
                return cls._read_txt(file_path)
            elif file_extension in ['.ply']:
                return cls._read_pcd(file_path)
            else:
                raise Exception('Unsupported file extension: %s' % file_extension)
        except Exception as e:
            print(f"Error occurred with file: {file_path}")
            print(f"Exception: {e}")

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        point_cloud = np.array(pc.points)
        return point_cloud


def compute_graph_nn_2(xyz, k_nn1, k_nn2):
    """
    Compute two KNN structures.
    :param xyz:
    :param k_nn1:
    :param k_nn2:
    :return:
    """
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    n_ver = xyz.shape[0]
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn2 + 1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    del nn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    # ---knn2---
    target2 = (neighbors.flatten()).astype('uint32')
    # ---knn1-----
    neighbors = neighbors[:, :k_nn1]
    distances = distances[:, :k_nn1]
    graph["source"] = np.tile(np.arange(0, n_ver), (k_nn1, 1)).flatten(order='F').astype('uint32')
    graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('uint32')
    graph["distances"] = distances.flatten().astype('float32')
    return graph, target2


def load_filenames(paths):
    """
    Load all `.npy` files under the `paths` directory.
    @param paths:
    @return:
    """
    files = []
    for path in paths:
        files.extend(path.glob("*.npy"))
    np.random.default_rng(seed=42).shuffle(files)
    return files


def load_labels(files, classes):
    """
    Obtain the label corresponding to each file.
    @param files:
    @param classes:
    @return:
    """
    labels = [classes.index(file.parents[1].name) for file in files]
    return labels
