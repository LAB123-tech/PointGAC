import time

from tqdm import tqdm

from datasets.dataset_utils.DatasetTransforms import Data_Augmentation
from utils import builder
from utils.AverageMeter import AverageMeter, Acc_Metric
from utils.loss import CrossEntropyLoss_Func as Loss_Func
from utils.main_util import *


def run_net(args, config, train_writer, val_writer, device, logger):
    # ------------------------------------------------------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------------------------------------------------------
    train_data_aug = Data_Augmentation(config.train_dataset.aug_list)
    val_data_aug = Data_Augmentation(config.val_dataset.aug_list)
    train_dataloader = builder.dataset_builder(config.train_dataset)
    val_dataloader = builder.dataset_builder(config.val_dataset)
    # ------------------------------------------------------------------------------------------------------------------
    # Construct scheduler
    # ------------------------------------------------------------------------------------------------------------------
    schedulers = builder.scheduler_builder(config.training.scheduler, train_dataloader)
    # ------------------------------------------------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------------------------------------------------
    model = builder.model_builder(config.model).to(device)
    # ------------------------------------------------------------------------------------------------------------------
    # Set loss function
    # ------------------------------------------------------------------------------------------------------------------
    loss_func = Loss_Func()
    # ------------------------------------------------------------------------------------------------------------------
    # Set optimizer
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
        val_metric = validate(model, val_dataloader, val_data_aug, config, epoch, loss_func, val_writer, logger, device)
        if val_metric.better_than(best_metrics):
            best_metrics = val_metric
            save_name = args.exp_name.split("finetune")[1][1:] + "-ckpt-best"
            save_checkpoint(model, optimizer, epoch, train_metric, best_metrics,
                            save_name, args, logger)
        elif val_metric.less_than(best_metrics):
            save_name = args.exp_name.split("finetune")[1][1:] + "-ckpt-last"
            save_checkpoint(model, optimizer, epoch, train_metric, val_metric,
                            save_name, args, logger)
        log_string(logger, f"------------------------------------------------------------------------------")
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
    Avg_loss = AverageMeter(['Loss_cls'])
    Avg_acc = AverageMeter(['Acc'])
    model.train()
    for idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training Progress"):
        # --------------------------------------------------------------------------------------------------------------
        # Update weights and learning rate; no weight decay for the first group
        # --------------------------------------------------------------------------------------------------------------
        iteration = len(train_dataloader) * epoch + idx
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = schedulers['lr'][iteration]
            if i == 1:
                param_group["weight_decay"] = schedulers['wd'][iteration]
        # --------------------------------------------------------------------------------------------------------------
        # Data augmentation
        # --------------------------------------------------------------------------------------------------------------
        points = data[0].float().to(device)
        points = train_data_aug(points)
        label = data[1].squeeze().to(device)
        # --------------------------------------------------------------------------------------------------------------
        # Forward pass
        # --------------------------------------------------------------------------------------------------------------
        optimizer.zero_grad()
        pred = model(points)
        loss_cls = loss_func(pred, label)
        Avg_loss.update([loss_cls.item()])
        # --------------------------------------------------------------------------------------------------------------
        # Calculate accuracy: number of correct predictions
        # --------------------------------------------------------------------------------------------------------------
        acc = (pred.argmax(-1) == label).sum() / float(label.size(0))
        Avg_acc.update([acc.item()], 0)
        # --------------------------------------------------------------------------------------------------------------
        # Gradient clipping to prevent explosion; controlled by max_norm parameter
        # --------------------------------------------------------------------------------------------------------------
        loss_cls.backward()
        if config.training.optimizer.clip_grad:
            clip_gradients(model, config.training.optimizer.clip_grad)
        optimizer.step()
    epoch_end_time = time.time()
    # ------------------------------------------------------------------------------------------------------------------
    # Record average loss and print
    # ------------------------------------------------------------------------------------------------------------------
    if train_writer is not None:
        train_writer.add_scalar('loss_cls', Avg_loss.avg(0), epoch)
        train_writer.add_scalar('Epoch/Acc', Avg_acc.avg(0), epoch)
    # ------------------------------------------------------------------------------------------------------------------
    # Log output
    # ------------------------------------------------------------------------------------------------------------------
    log_string(logger, '[Training] EPOCH: %d EpochTime = %.3f (s) Loss_cls = %.4f Acc = %s' %
               (epoch, epoch_end_time - epoch_start_time, Avg_loss.avg()[0], Avg_acc.avg(0)))
    return Acc_Metric(Avg_acc.avg(0))


def validate(model, val_dataloader, val_data_aug, config, epoch, loss_func, val_writer, logger, device):
    epoch_start_time = time.time()
    Avg_loss = AverageMeter(['Loss_cls'])
    Avg_acc = AverageMeter(['Acc'])
    model.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validation Progress"):
            # ----------------------------------------------------------------------------------------------------------
            # Data augmentation
            # ----------------------------------------------------------------------------------------------------------
            points = data[0].float().to(device)
            points = val_data_aug(points)
            label = data[1].squeeze().to(device)
            # ----------------------------------------------------------------------------------------------------------
            # Forward pass
            # ----------------------------------------------------------------------------------------------------------
            pred = model(points)
            loss_cls = loss_func(pred, label)
            Avg_loss.update([loss_cls.item()])
            # ----------------------------------------------------------------------------------------------------------
            # Calculate accuracy: number of correct predictions
            # ----------------------------------------------------------------------------------------------------------
            acc = (pred.argmax(-1) == label).sum() / float(label.size(0))
            Avg_acc.update([acc.item()], 0)
    epoch_end_time = time.time()
    # ------------------------------------------------------------------------------------------------------------------
    # Record average loss and print
    # ------------------------------------------------------------------------------------------------------------------
    if val_writer is not None:
        val_writer.add_scalar('loss_cls', Avg_loss.avg(0), epoch)
        val_writer.add_scalar('Epoch/Acc', Avg_acc.avg(0), epoch)
    # ------------------------------------------------------------------------------------------------------------------
    # Log output
    # ------------------------------------------------------------------------------------------------------------------
    log_string(logger, '[Validation] EPOCH: %d EpochTime = %.3f (s) Loss_cls = %.4f Acc = %s' %
               (epoch, epoch_end_time - epoch_start_time, Avg_loss.avg()[0], Avg_acc.avg(0)))
    return Acc_Metric(Avg_acc.avg(0))
