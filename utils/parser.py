import os
import argparse
from utils.main_util import create_experiment_dir


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='GPU device index')
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--loss', type=str, default='smooth_l1', help='loss name')
    parser.add_argument('--save_ckpts', type=str, default="checkpoints", help='path to save ckpt')
    parser.add_argument('--way', type=int, default=-1, help="class number, 5 or 10")
    parser.add_argument('--shot', type=int, default=-1, help="training object number, 10 or 20")
    parser.add_argument('--fold', type=int, default=-1, help="fold number, max=10")
    parser.add_argument('--resume', action='store_true', default=False, help='interrupted by accident')
    parser.add_argument('--finetune_model_cls', action='store_true', help='finetune classification')
    parser.add_argument('--finetune_model_seg', action='store_true', help='finetune segmentation')

    args = parser.parse_args()

    args.experiment_path = os.path.join('./experiments', args.exp_name)
    args.tfboard_path = os.path.join(args.experiment_path, 'tfboard')
    args.log_path = os.path.join(args.experiment_path, 'logs')
    args.save_ckpts = os.path.join(str(args.save_ckpts), args.exp_name)

    create_experiment_dir(args)
    return args
