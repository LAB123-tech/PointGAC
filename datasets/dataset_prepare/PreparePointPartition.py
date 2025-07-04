import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from datasets.dataset_utils.DatasetFunc import *

sys.path.append("../lib/Partition_lib/cut-pursuit/build/src")
sys.path.append("../lib/Partition_lib/ply_c")
import libcp
import libply_c


def process_point_cloud(data_path):
    point_xyz = np.load(data_path)[:, :3].astype('float32')
    # ------------------------------------------------------------------------------------------------------------------
    # Perform homogeneous geometric partitioning on the point cloud to obtain geometric classification results.
    # ------------------------------------------------------------------------------------------------------------------
    graph_nn, target_fea = compute_graph_nn_2(point_xyz, 10, 30)
    graph_nn["edge_weight"] = 1. / (1.0 + graph_nn["distances"] / np.mean(graph_nn["distances"]))
    # ------------------------------------------------------------------------------------------------------------------
    # Compute geometric features.
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud_geo = libply_c.compute_geof(point_xyz, target_fea, 30).astype('float32')
    point_cloud_geo[:, 3] = 2. * point_cloud_geo[:, 3]
    # ------------------------------------------------------------------------------------------------------------------
    # Perform L0 Cut segmentation algorithm.
    # ------------------------------------------------------------------------------------------------------------------
    components, point_partition = libcp.cutpursuit(point_cloud_geo,
                                                   graph_nn["source"],
                                                   graph_nn["target"],
                                                   graph_nn["edge_weight"],
                                                   0.08)
    # ------------------------------------------------------------------------------------------------------------------
    # Add the segmentation results as a new dimension.
    # ------------------------------------------------------------------------------------------------------------------
    point_partition = np.array(point_partition).astype('int64').reshape(-1, 1)
    point_data = np.concatenate((point_xyz, point_partition), axis=-1)
    # ------------------------------------------------------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------------------------------------------------------
    np.save(data_path, point_data)
    return True


def main(args):
    data_dir = Path(args.src_dir)
    meshes_paths = list(data_dir.glob(f'**/*.{args.ext}'))
    # ------------------------------------------------------------------------------------------------------------------
    # Process each point cloud in parallel.
    # ------------------------------------------------------------------------------------------------------------------
    if args.n_process > 0:
        p = Pool(args.n_process)
        results = []
        for result in tqdm(p.imap_unordered(process_point_cloud, meshes_paths), total=len(meshes_paths)):
            results.append(result)
    else:
        results = [process_point_cloud(mesh_path) for mesh_path in tqdm(meshes_paths)]
    print(f"Done. {sum(results)} files were created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--src_dir", type=str, default="/home/x3022/文档/lab/data/ModelNet40-my/",
    #                     help="Directory containing point cloud files.")
    parser.add_argument("--src_dir", type=str, default="/home/x3022/文档/lab/data/ShapeNet-my/",
                        help="Directory containing point cloud files.")
    parser.add_argument("--n_process", type=int, default=0,
                        help="Number of parallel processes.")
    parser.add_argument("--ext", type=str, default="npy",
                        help="processed data type.")
    args = parser.parse_args()
    main(args)
