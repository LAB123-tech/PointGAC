import time

import numpy as np
import torch.nn as nn
from tqdm import tqdm

from datasets.dataset_utils.DatasetTransforms import Data_Augmentation
from utils import builder
from utils import main_util
from utils.Accuracy import *
from utils.AverageMeter import AverageMeter, Acc_Metric
from utils.loss import CrossEntropyLoss_Func as Loss_Func
from utils.main_util import *

loss_ce = nn.CrossEntropyLoss()


def run_net(args, config, train_writer, val_writer, device, logger):
    # ------------------------------------------------------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------------------------------------------------------
    train_data_aug = Data_Augmentation(config.train_dataset.aug_list)
    val_data_aug = Data_Augmentation(config.val_dataset.aug_list)
    train_dataloader = builder.dataset_builder(config.train_dataset)
    val_dataloader = builder.dataset_builder(config.val_dataset)
    # ------------------------------------------------------------------------------------------------------------------
    # Build schedulers
    # ------------------------------------------------------------------------------------------------------------------
    schedulers = builder.scheduler_builder(config.training.scheduler, train_dataloader)
    # ------------------------------------------------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------------------------------------------------
    model = builder.model_builder(config.model).to(device)
    # ------------------------------------------------------------------------------------------------------------------
    # Setup loss function
    # ------------------------------------------------------------------------------------------------------------------
    loss_func = Loss_Func()
    # ------------------------------------------------------------------------------------------------------------------
    # Setup optimizer
    # ------------------------------------------------------------------------------------------------------------------
    optimizer = builder.optimizer_builder(config.training.optimizer, model)
    # ------------------------------------------------------------------------------------------------------------------
    # Load weights for fine-tuning
    # ------------------------------------------------------------------------------------------------------------------
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    load_model_to_finetune(model, logger)
    # ------------------------------------------------------------------------------------------------------------------
    # Start training
    # ------------------------------------------------------------------------------------------------------------------
    for epoch in range(start_epoch, config.training.scheduler.epochs + 1):
        train_metric = train(model, train_dataloader, train_data_aug, config, epoch, optimizer,
                             loss_func, train_writer, schedulers, logger, device)
        val_metric = validate(model, val_dataloader, val_data_aug, epoch, loss_func, val_writer, config, logger, device)
        if val_metric.better_than(best_metrics):
            best_metrics = val_metric
            save_name = args.exp_name.split("finetune")[1][1:] + "-ckpt-best"
            save_checkpoint(model, optimizer, epoch, train_metric, best_metrics,
                            save_name, args, logger)
        elif val_metric.less_than(best_metrics):
            save_name = args.exp_name.split("finetune")[1][1:] + "-ckpt-last"
            save_checkpoint(model, optimizer, epoch, train_metric, val_metric,
                            save_name, args, logger)
        log_string(logger, f"---------------------------------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------
    # Close tensorboard writers
    # ------------------------------------------------------------------------------------------------------------------
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def train(model, train_dataloader, train_data_aug, config, epoch, optimizer, loss_func, train_writer, schedulers,
          logger, device):
    epoch_start_time = time.time()
    Avg_loss = AverageMeter(['Loss_seg'])
    Avg_acc_ins_cls_iou = AverageMeter(['Acc', 'ins_iou', 'cls_iou'])
    every_cls_iou = {single_class: [] for single_class in seg_classes.keys()}
    model.train()
    for idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training Progress"):
        # --------------------------------------------------------------------------------------------------------------
        # Update weights and learning rate, no weight decay for first group
        # --------------------------------------------------------------------------------------------------------------
        iteration = len(train_dataloader) * epoch + idx
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = schedulers['lr'][iteration]
            if i == 1:
                param_group["weight_decay"] = schedulers['wd'][iteration]
        # --------------------------------------------------------------------------------------------------------------
        # Data augmentation
        # --------------------------------------------------------------------------------------------------------------
        point_data = data[0].float().to(device)
        point_data = train_data_aug(point_data)
        cls_label = data[1].long().to(device)
        seg_label = point_data[:, :, -1].long().to(device)
        # --------------------------------------------------------------------------------------------------------------
        # Forward pass
        # --------------------------------------------------------------------------------------------------------------
        optimizer.zero_grad()
        seg_pred = model(point_data[:, :, :3], main_util.cls2onehot(cls_label, config.model.label.cls_dim, device))
        loss_seg = loss_func(seg_pred.reshape(-1, config.model.seg_head.part_dim), seg_label.view(-1, ))
        Avg_loss.update([loss_seg.item()])
        # --------------------------------------------------------------------------------------------------------------
        # Calculate accuracy, class_iou, and instance_iou
        # --------------------------------------------------------------------------------------------------------------
        pred_choice = cal_pred_choice(seg_pred, seg_label)
        Avg_acc_ins_cls_iou = cal_acc(pred_choice, seg_label, Avg_acc_ins_cls_iou)
        Avg_acc_ins_cls_iou, every_cls_iou = cal_ins_cls_iou(pred_choice, seg_label, Avg_acc_ins_cls_iou, every_cls_iou)
        # --------------------------------------------------------------------------------------------------------------
        # Gradient clipping to prevent exploding gradients, controlled by max_norm parameter
        # --------------------------------------------------------------------------------------------------------------
        loss_seg.backward()
        if config.training.optimizer.clip_grad:
            clip_gradients(model, config.training.optimizer.clip_grad)
        optimizer.step()
    epoch_end_time = time.time()
    # ------------------------------------------------------------------------------------------------------------------
    # Calculate average iou for each class and then overall average iou
    # ------------------------------------------------------------------------------------------------------------------
    for class_name in every_cls_iou.keys():
        every_cls_iou[class_name] = np.mean(every_cls_iou[class_name])
        Avg_acc_ins_cls_iou.update(np.mean(every_cls_iou[class_name]), 2)
    # ------------------------------------------------------------------------------------------------------------------
    # Record average loss and print
    # ------------------------------------------------------------------------------------------------------------------
    if train_writer is not None:
        train_writer.add_scalar('loss_seg', Avg_loss.avg(0), epoch)
        train_writer.add_scalar('Acc', Avg_acc_ins_cls_iou.avg(0), epoch)
        train_writer.add_scalar('ins_iou', Avg_acc_ins_cls_iou.avg(1), epoch)
        train_writer.add_scalar('cls_iou', Avg_acc_ins_cls_iou.avg(2), epoch)
    log_string(logger, '[Training] EPOCH: %d EpochTime = %.3f (s) Loss_seg = %.4f, Acc = %s ins_iou = %s cls_iou = %s' %
               (epoch, epoch_end_time - epoch_start_time,
                Avg_loss.avg()[0], Avg_acc_ins_cls_iou.avg(0), Avg_acc_ins_cls_iou.avg(1), Avg_acc_ins_cls_iou.avg(2)))
    return Acc_Metric(Avg_acc_ins_cls_iou.avg(1))


def validate(model, val_dataloader, val_data_aug, epoch, loss_func, val_writer, config, logger, device):
    epoch_start_time = time.time()
    Avg_loss = AverageMeter(['Loss_seg'])
    Avg_acc_ins_cls_iou = AverageMeter(['Acc', 'ins_iou', 'cls_iou'])
    every_cls_iou = {single_class: [] for single_class in seg_classes.keys()}
    model.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validation Progress"):
            # ----------------------------------------------------------------------------------------------------------
            # Data augmentation
            # ----------------------------------------------------------------------------------------------------------
            point_data = data[0].float().to(device)
            point_data = val_data_aug(point_data)
            cls_label = data[1].long().to(device)
            seg_label = point_data[:, :, -1].long().to(device)
            # ----------------------------------------------------------------------------------------------------------
            # Forward pass
            # ----------------------------------------------------------------------------------------------------------
            seg_pred = model(point_data[:, :, :3], main_util.cls2onehot(cls_label, config.model.label.cls_dim, device))
            loss_seg = loss_func(seg_pred.reshape(-1, config.model.seg_head.part_dim), seg_label.view(-1, ))
            Avg_loss.update([loss_seg.item()])
            # ----------------------------------------------------------------------------------------------------------
            # Calculate accuracy, class_iou, and instance_iou
            # ----------------------------------------------------------------------------------------------------------
            pred_choice = cal_pred_choice(seg_pred, seg_label)
            Avg_acc_ins_cls_iou = cal_acc(pred_choice, seg_label, Avg_acc_ins_cls_iou)
            Avg_acc_ins_cls_iou, every_cls_iou = cal_ins_cls_iou(pred_choice, seg_label, Avg_acc_ins_cls_iou,
                                                                 every_cls_iou)
    epoch_end_time = time.time()
    # ------------------------------------------------------------------------------------------------------------------
    # Calculate average iou for each class and then overall average iou
    # ------------------------------------------------------------------------------------------------------------------
    for class_name in every_cls_iou.keys():
        every_cls_iou[class_name] = np.mean(every_cls_iou[class_name])
        Avg_acc_ins_cls_iou.update(np.mean(every_cls_iou[class_name]), 2)
    # ------------------------------------------------------------------------------------------------------------------
    # Record average loss and print
    # ------------------------------------------------------------------------------------------------------------------
    if val_writer is not None:
        val_writer.add_scalar('loss_seg', Avg_loss.avg(0), epoch)
        val_writer.add_scalar('Acc', Avg_acc_ins_cls_iou.avg(0), epoch)
        val_writer.add_scalar('ins_iou', Avg_acc_ins_cls_iou.avg(1), epoch)
        val_writer.add_scalar('cls_iou', Avg_acc_ins_cls_iou.avg(2), epoch)
    log_string(logger,
               '[Validation] EPOCH: %d EpochTime = %.3f (s) Loss_seg = %.4f, Acc = %s ins_iou = %s cls_iou = %s' %
               (epoch, epoch_end_time - epoch_start_time,
                Avg_loss.avg()[0], Avg_acc_ins_cls_iou.avg(0), Avg_acc_ins_cls_iou.avg(1), Avg_acc_ins_cls_iou.avg(2)))
    return Acc_Metric(Avg_acc_ins_cls_iou.avg(1))
