import shutil

import torch
from tensorboardX import SummaryWriter

from tools.runner_finetune_cls import run_net as finetune_cls
from tools.runner_finetune_seg import run_net as finetune_seg
from tools.runner_pretrain import run_net as pretrain
from utils import parser, main_util
from utils.config import *
from utils.logger import *


def main():
    # ------------------------------------------------------------------------------------------------------------------
    # Set arguments and CUDA device
    # ------------------------------------------------------------------------------------------------------------------
    args = parser.get_args()
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    # ------------------------------------------------------------------------------------------------------------------
    # Setup logger
    # ------------------------------------------------------------------------------------------------------------------
    logger = create_logger(args.log_path, args.exp_name)
    # ------------------------------------------------------------------------------------------------------------------
    # Remove old TensorBoard log directory and create new folders
    # Make sure to run from the source directory
    # tensorboard --logdir=experiments/pretrain/tfboard/train
    # ------------------------------------------------------------------------------------------------------------------
    if os.path.exists(args.tfboard_path):
        shutil.rmtree(args.tfboard_path)
        log_string(logger, f'[Board] Deleted existing TensorBoard log directory: {args.tfboard_path}')
    os.makedirs(os.path.join(args.tfboard_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.tfboard_path, 'val'), exist_ok=True)
    train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
    val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'val'))
    # ------------------------------------------------------------------------------------------------------------------
    # Load config file
    # ------------------------------------------------------------------------------------------------------------------
    config = cfg_from_yaml_file(args.config)
    # ------------------------------------------------------------------------------------------------------------------
    # Set random seed
    # ------------------------------------------------------------------------------------------------------------------
    if args.seed is not None:
        log_string(logger, f'[Random] Set random seed to {args.seed}')
        main_util.set_random_seed(args.seed)
    # ------------------------------------------------------------------------------------------------------------------
    # Set few-shot learning parameters
    # ------------------------------------------------------------------------------------------------------------------
    if args.shot != -1:
        config.train_dataset.others.shot = args.shot
        config.train_dataset.others.way = args.way
        config.train_dataset.others.fold = args.fold
        config.val_dataset.others.shot = args.shot
        config.val_dataset.others.way = args.way
        config.val_dataset.others.fold = args.fold
    # ------------------------------------------------------------------------------------------------------------------
    # Pretrain and finetune logic
    # ------------------------------------------------------------------------------------------------------------------
    if args.finetune_model_cls:
        finetune_cls(args, config, train_writer, val_writer, device, logger)
    if args.finetune_model_seg:
        finetune_seg(args, config, train_writer, val_writer, device, logger)
    else:
        pretrain(args, config, train_writer, val_writer, device, logger)


if __name__ == '__main__':
    main()
