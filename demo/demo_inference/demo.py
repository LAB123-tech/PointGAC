import argparse

import numpy as np
import open3d as o3d
from tqdm import tqdm

# 这句要保留，完成模型的注册
# from tools import *
from utils import builder
from utils.config import *
from utils.main_util import *
from utils.visualize import visualize_pca_and_save


def farthest_point_sample(points, n_samples):
    """
    最远点采样
    :param points: 点云数据 [N, 3]
    :param n_samples: 采样点数
    :return: 采样后的点云 [n_samples, 3]
    """
    points = np.array(points)
    N = points.shape[0]
    sample_indices = np.zeros(n_samples, dtype=np.int32)
    distances = np.ones(N) * 1e10

    # 随机选择第一个点
    first_idx = np.random.randint(0, N)
    sample_indices[0] = first_idx

    for i in range(1, n_samples):
        last_selected = points[sample_indices[i - 1]]
        dist_to_last = np.sum((points - last_selected) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)
        sample_indices[i] = np.argmax(distances)

    return points[sample_indices]


def normalize_point_cloud(points):
    """
    点云归一化
    :param points: 点云数据 [N, 3]
    :return: 归一化后的点云 [N, 3]
    """
    # 中心化
    centroid = np.mean(points, axis=0)
    points -= centroid

    # 缩放
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points /= max_dist

    return points


def process_ply_files(ply_folder, model, device, num_points=1024):
    ply_files = [f for f in os.listdir(ply_folder) if f.endswith('.ply')]

    model.eval()
    with torch.no_grad():
        for idx, ply_file in enumerate(tqdm(ply_files, desc="Processing PLY files")):
            # 1. Read PLY files
            ply_path = os.path.join(ply_folder, ply_file)
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)

            # 2. Farthest Point Sampling (FPS)
            if len(points) > num_points:
                points = farthest_point_sample(points, num_points)
            elif len(points) < num_points:
                # If the number of points is insufficient, repeat sampling.
                indices = np.random.choice(len(points), num_points - len(points), replace=True)
                points = np.concatenate([points, points[indices]], axis=0)

            # 3. Normalization
            points = normalize_point_cloud(points)

            # 4. Convert to torch tensor and add batch dimension
            points = torch.from_numpy(points).float().unsqueeze(0).to(device)  # [1, N, 3]

            # 5. Model inference
            feature, center = model.forward_test(points[:, :, :3])

            # 6. PCA visualization
            visualize_pca_and_save(points[0][:, :3], feature[0], center[0], idx)


if __name__ == "__main__":
    # Parameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_folder', type=str, default="data",
                        help='Path to folder containing PLY files')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points to sample')
    parser.add_argument('--model_path', type=str, default="../../checkpoints/pretrain/pretrain-ckpt-best.pth",
                        help='Path to trained model weights')
    parser.add_argument('--config_file', type=str, default="../../cfgs/demo_config/demo.yaml",
                        help='Number of points to sample')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    # Ensure the PLY folder exists
    if not os.path.isdir(args.ply_folder):
        raise ValueError(f"PLY folder not found: {args.ply_folder}")

    # Prepare configuration and model; here only the student model is initialized,
    # the teacher model should be set via model.set_teacher()
    config = cfg_from_yaml_file(args.config_file)
    device = torch.device(args.device)
    model = builder.model_builder(config.model).to(device)

    # Load student model weights
    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded model weights from {args.model_path}")
    else:
        raise ValueError(f"Model weights not found at {args.model_path}")

    # Process PLY files
    process_ply_files(args.ply_folder, model, device, args.num_points)
