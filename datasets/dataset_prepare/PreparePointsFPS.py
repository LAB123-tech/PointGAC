import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.ops import sample_farthest_points
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument("--in_dir", type=str, default="/media/lab/Sandi/DatasetPointCloud/ModelNet40/ModelNet40-off")
# parser.add_argument("--out_dir", type=str, default="/media/lab/Sandi/DatasetPointCloud/ModelNet40/ModelNet40-my")
# parser.add_argument("--ext", type=str, default="off")
parser.add_argument("--in_dir", type=str, default="/media/lab/Sandi/DatasetPointCloud/ShapeNet/ShapeNet-obj/04554684")
parser.add_argument("--out_dir", type=str, default="/media/lab/Sandi/DatasetPointCloud/ShapeNet/ShapeNet-my/04554684")
parser.add_argument("--ext", type=str, default="obj")
parser.add_argument("--n_points", type=int, default=8192)
parser.add_argument("--force", action="store_true", default=False)
parser.add_argument("--n_process", type=int, default=16)
args = parser.parse_args()

input_path = Path(args.in_dir)
output_path = Path(args.out_dir)
meshes_paths = list(input_path.glob(f'**/*.{args.ext}'))


def sample_point_cloud(mesh_path):
    new_model_path = mesh_path.parent / (mesh_path.stem + ".npy")
    new_model_path = output_path / new_model_path.relative_to(input_path)
    new_model_path.parent.mkdir(exist_ok=True, parents=True)

    if not new_model_path.exists() or args.force:
        mesh = trimesh.load(mesh_path, force="mesh")
        point_cloud = mesh.sample(min(args.n_points * 8, 16384))
        point_cloud = torch.from_numpy(point_cloud).float().unsqueeze(0)
        point_cloud = sample_farthest_points(point_cloud, K=args.n_points, random_start_point=True)[0]
        point_cloud = point_cloud.squeeze().numpy()
        np.save(new_model_path, point_cloud)
        return True
    return False


if args.n_process > 0:
    p = Pool(args.n_process)
    results = []
    for result in tqdm(p.imap_unordered(sample_point_cloud, meshes_paths), total=len(meshes_paths)):
        results.append(result)
else:
    results = [sample_point_cloud(mesh_path) for mesh_path in tqdm(meshes_paths)]

print(f"Done. {sum(results)} files were created.")
