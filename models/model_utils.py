import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_farthest_points, knn_gather
from pytorch3d.ops.utils import masked_gather


# from utils.visualize import vis_points_with_label

def square_distance(src, dst):
    """
    Calculate the squared distance from each point in the source point cloud to all points in the target point cloud.
    :param src: source points, (B, N, C)
    :param dst: target points, (B, M, C)
    :return: squared distances between each point, (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def mask_center_rand(point_data, mask_ratio):
    """
    Generate random mask positions.
    :param point_data: B G 3
    :param mask_ratio: float
    :return: mask : B G (bool)
    """
    B, G, _ = point_data.shape
    # Regions with value 1 indicate the areas to be masked
    num_mask = int(mask_ratio * G)
    mask = np.hstack([np.zeros(G - num_mask, dtype=bool),
                      np.ones(num_mask, dtype=bool)])
    # Independently shuffle each row
    overall_mask = np.tile(mask, (B, 1))
    random_indices = np.argsort(np.random.rand(B, G), axis=1)
    overall_mask = np.take_along_axis(overall_mask, random_indices, axis=1)
    overall_mask = torch.from_numpy(overall_mask).to(point_data.device)
    return overall_mask.int()


def mask_center_block(point_data, mask_ratio):
    """
    Generate block-wise mask positions.
    :param point_data: B G 3
    :param mask_ratio: float
    :return: mask : B G (bool)
    """
    B, G, _ = point_data.shape
    if mask_ratio == 0:
        return torch.zeros(point_data.shape[:2], device=point_data.device).bool()
    # Get center indices for all batches
    center_indices = torch.randint(0, G, (B,), device=point_data.device)
    center_points = point_data[torch.arange(B), center_indices]  # B x 3
    # Compute distance matrix between center points and all points, mask the closest mask_ratio * G points
    distance_matrix = torch.norm(point_data - center_points[:, None, :], p=2, dim=-1)  # B x G
    sorted_indices = torch.argsort(distance_matrix, dim=-1)  # B x G
    mask_num = int(mask_ratio * G)
    # Generate mask for contiguous regions
    mask = torch.zeros((B, G), device=point_data.device)
    mask[torch.arange(B).unsqueeze(1), sorted_indices[:, :mask_num]] = 1
    return mask.bool()


def point_cloud_grouping(points, num_groups, group_size):
    """
    Group the point cloud.
    @param points: (B, N, C)
    @param num_groups: number of groups, G
    @param group_size: size of each group, k
    @return:
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Sample farthest points as group centers
    # ------------------------------------------------------------------------------------------------------------------
    _, idx = sample_farthest_points(points[:, :, :3].float(), K=num_groups, random_start_point=True)
    group_centers = masked_gather(points, idx)  # (B, G, C)
    # ------------------------------------------------------------------------------------------------------------------
    # Group points using KNN
    # ------------------------------------------------------------------------------------------------------------------
    _, idx, _ = knn_points(group_centers.float(),
                           points[:, :, :3].float(),
                           K=group_size,
                           return_sorted=False,
                           return_nn=False)
    group_data = knn_gather(points, idx)  # (B, G, K, C)
    # ------------------------------------------------------------------------------------------------------------------
    # Convert point coordinates to relative coordinates w.r.t. group centers
    # ------------------------------------------------------------------------------------------------------------------
    group_data[:, :, :, :3] = group_data[:, :, :, :3] - group_centers.unsqueeze(2)
    return group_data, group_centers


def sinkhorn_knopp(r, c, M, reg=1e-3, error_min=1e-5, num_iters=100, mask=None, device=None):
    """
    Batch sinkhorn iteration.
    @param r: tensor with shape (n, d1), the first distribution .
    @param c: tensor with shape (n, d2), the second distribution.
    @param M: tensor with shape (n, d1, d2) the cost metric.
    @param reg: factor for entropy regularization.
    @param error_min: the error threshold to stop the iteration.
    @param num_iters: number of total iterations.
    @param mask: mask region
    @param device: cuda or cpu
    @return:
    """
    n, d1, d2 = M.shape
    assert r.shape[0] == c.shape[0] == n and r.shape[1] == d1 and c.shape[1] == d2, \
        'r.shape=%s, v.shape=%s, M.shape=%s' % (r.shape, c.shape, M.shape)
    if mask is None:
        mask = torch.ones_like(M)

    K = (-M / reg).exp()  # (n, d1, d2)
    u = torch.ones_like(r, device=device) / d1  # (n, d1)
    v = torch.ones_like(c, device=device) / d2  # (n, d2)

    for _ in range(num_iters):
        r0 = u
        v_full = v[:, None, ].repeat(1, d1, 1)
        u = r / (torch.sum(K * v_full * mask, dim=-1) + 1e-5)
        u_full = u[:, :, None].repeat(1, 1, d2)
        v = c / (torch.sum(K * u_full * mask, dim=1) + 1e-5)
        err = (u - r0).abs().mean()
        if err.item() < error_min:
            break

    T = torch.einsum('ij,ik->ijk', [u, v]) * K * mask
    return T


def ot_assign(ori_xyz, sample_xyz, ori_partition, sample_partition, mask):
    """
    Use optimal transport to assign each point to its corresponding cluster.
    @param ori_xyz: coordinates of the original point cloud
    @param sample_xyz: coordinates of the sampled point cloud
    @param ori_partition: geometric partition labels of the original point cloud
    @param sample_partition: geometric partition labels of the sampled point cloud
    @param mask: region mask
    @return:
    """
    device = ori_xyz.device
    batch_size, num_ori, dim = ori_xyz.shape
    num_sample = sample_xyz.shape[1]
    # ------------------------------------------------------------------------------------------------------------------
    # Get the number of points per category in the original point cloud,
    # and assign the corresponding counts to each point's label
    # ------------------------------------------------------------------------------------------------------------------
    max_labels_ori = torch.max(ori_partition, dim=1,
                               keepdim=True).values + 1  # Count number of labels per batch, shape (b, 1)
    offsets_ori = torch.cumsum(max_labels_ori, dim=0) - max_labels_ori  # Offsets for label indexing, shape (b, 1)
    offset_ori_partition = ori_partition + offsets_ori  # Apply offsets to partition labels, shape (b, n_points)
    num_label_ori = torch.bincount(offset_ori_partition.flatten())  # Count points per label
    num_class_ori = num_label_ori[offset_ori_partition.flatten()].reshape(batch_size, -1)  # Counts per point's label
    # ------------------------------------------------------------------------------------------------------------------
    # Get the number of points per category in the sampled point cloud,
    # and assign the corresponding counts to each point's label
    # ------------------------------------------------------------------------------------------------------------------
    max_labels_sample = torch.max(sample_partition, dim=1, keepdim=True).values + 1
    offsets_sample = torch.cumsum(max_labels_sample, dim=0) - max_labels_sample
    offset_sample_partition = sample_partition + offsets_sample
    num_label_sample = torch.bincount(offset_sample_partition.flatten())
    num_class_sample = num_label_sample[offset_sample_partition.flatten()].reshape(batch_size, -1)
    # ------------------------------------------------------------------------------------------------------------------
    # Initialize uniform distributions 1/n, where n is number of geometric sub-blocks
    # ------------------------------------------------------------------------------------------------------------------
    r = torch.ones(batch_size, num_ori, device=device) / num_class_ori
    c = torch.ones(batch_size, num_sample, device=device) / num_class_sample
    # ----------------------------------------------------------------------------------------------------------------------
    # Use Sinkhorn-Knopp algorithm to achieve balanced partitioning within sub-blocks
    # ----------------------------------------------------------------------------------------------------------------------
    cost_xyz = square_distance(ori_xyz, sample_xyz)
    T = sinkhorn_knopp(r, c, cost_xyz, mask=mask, device=device)
    return T, offset_sample_partition


def cluster_ot(ori_data, num_clusters):
    """
    Perform clustering using Optimal Transport (OT), and normalize each point by subtracting its cluster center.
    @param ori_data: original data tensor of shape (B, N, C)
    @param num_clusters: number of clusters
    @return:
    """
    B, N, _ = ori_data.shape
    _, idx = sample_farthest_points(ori_data.float(), K=num_clusters, random_start_point=True)
    sample_data = masked_gather(ori_data, idx)
    # ------------------------------------------------------------------------------------------------------------------
    # Generate mask
    # ------------------------------------------------------------------------------------------------------------------
    ori_partition = ori_data[:, :, 3].long()
    sample_partition = sample_data[:, :, 3].long()
    ori_labels_1024 = ori_partition[:, :, None].repeat(1, 1, num_clusters)
    center_labels_128 = sample_partition[:, None, :].repeat(1, N, 1)
    mask = (ori_labels_1024 == center_labels_128).int()
    # ------------------------------------------------------------------------------------------------------------------
    # Use OT to assign points to clusters
    # ------------------------------------------------------------------------------------------------------------------
    assign_matrix, sample_partition = ot_assign(ori_data[:, :, :3], sample_data[:, :, :3],
                                                ori_partition, sample_partition, mask)
    choice = torch.argmax(assign_matrix, dim=-1)
    # ------------------------------------------------------------------------------------------------------------------
    # Visualization (optional)
    # ------------------------------------------------------------------------------------------------------------------
    # for i in range(B):
    #     vis_points_with_label(ori_data[i, :, :3].detach().cpu().numpy(), ori_data[i, :, -1].detach().cpu().numpy(),
    #                           "L0 Cut-" + str(i))
    #     vis_points_with_label(ori_data[i, :, :3].detach().cpu().numpy(), choice[i].detach().cpu().numpy(),
    #                           "OT-" + str(i))
    # ------------------------------------------------------------------------------------------------------------------
    # Compute cluster center for each point in the original point cloud (shape same as ori_data)
    # ------------------------------------------------------------------------------------------------------------------
    cluster_center = sample_data.gather(1, choice.unsqueeze(-1).expand(-1, -1, ori_data.shape[-1]))
    # ------------------------------------------------------------------------------------------------------------------
    # Normalize each point by subtracting its cluster center
    # ------------------------------------------------------------------------------------------------------------------
    normalized_xyz = ori_data[:, :, :3] - cluster_center[:, :, :3]
    # ------------------------------------------------------------------------------------------------------------------
    # Update sample_data's geometric labels to be unique per point cloud
    # ------------------------------------------------------------------------------------------------------------------
    sample_data[:, :, -1] = sample_partition
    return sample_data, normalized_xyz, choice


def is_float_dtype(dtype):
    """
    Check if the given dtype is a floating point type
    @param dtype:
    @return:
    """

    return any([dtype == float_dtype
                for float_dtype in (torch.float64,
                                    torch.float32,
                                    torch.float16,
                                    torch.bfloat16)
                ])


def sort_by_labels(features, labels):
    """
    Reorder features and labels according to the labels
    @param features:
    @param labels:
    @return:
    """
    sorted_indices = torch.argsort(labels)
    sorted_features = features[sorted_indices]
    sorted_labels = labels[sorted_indices]
    return sorted_features, sorted_labels


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def CorrelationPearson(src, dst):
    """
    Calculate the correlation coefficient
    @param src: [B, N, C]
    @param dst: [B, N, C]
    @return: [B, N, N]
    """
    channel_src_mean = torch.mean(src, dim=-1)
    channel_dst_mean = torch.mean(dst, dim=-1)
    src = src - channel_src_mean[:, :, None]
    dst = dst - channel_dst_mean[:, :, None]
    conv_affinity = torch.matmul(src, dst.permute(0, 2, 1))
    Standard_deviation_x = torch.sum(src * src, dim=-1)
    Standard_deviation_y = torch.sum(dst * dst, dim=-1)
    result = conv_affinity / torch.sqrt(
        torch.matmul(Standard_deviation_x[:, :, None], Standard_deviation_y[:, :, None].permute(0, 2, 1)))
    return result
