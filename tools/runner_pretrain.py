import time

from tqdm import tqdm

from datasets.dataset_utils.DatasetTransforms import Data_Augmentation
from utils import builder
from utils.AverageMeter import AverageMeter, Acc_Metric
from utils.loss import KL_Loss_Func as Loss_Func
from utils.main_util import *
from utils.visualize import visualize_distribution, visualize_tsne


def run_net(args, config, train_writer, val_writer, device, logger):
    # ------------------------------------------------------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------------------------------------------------------
    train_data_aug = Data_Augmentation(config.train_dataset.aug_list)
    extra_train_aug = Data_Augmentation(config.extra_train_dataset.aug_list)
    val_data_aug = Data_Augmentation(config.val_dataset.aug_list)
    train_dataloader = builder.dataset_builder(config.train_dataset)
    extra_train_dataloader = builder.dataset_builder(config.extra_train_dataset)
    val_dataloader = builder.dataset_builder(config.val_dataset)
    # ------------------------------------------------------------------------------------------------------------------
    # Build schedulers
    # ------------------------------------------------------------------------------------------------------------------
    schedulers = builder.scheduler_builder(config.training.scheduler, train_dataloader)
    # ------------------------------------------------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------------------------------------------------
    model = builder.model_builder(config.model).to(device)
    model.set_teacher()
    # ------------------------------------------------------------------------------------------------------------------
    # Setup loss function
    # ------------------------------------------------------------------------------------------------------------------
    loss_func = Loss_Func()
    # ------------------------------------------------------------------------------------------------------------------
    # Setup optimizer
    # ------------------------------------------------------------------------------------------------------------------
    optimizer = builder.optimizer_builder(config.training.optimizer, model)
    # ------------------------------------------------------------------------------------------------------------------
    # Load checkpoint and resume training if specified
    # ------------------------------------------------------------------------------------------------------------------
    if args.resume:
        start_epoch, best_metric = load_model(args, logger, is_train=True, model=model, optimizer=optimizer)
        best_metrics = Acc_Metric(best_metric)
    else:
        start_epoch = 0
        best_metrics = Acc_Metric(0.0)
    # ------------------------------------------------------------------------------------------------------------------
    # Start training
    # ------------------------------------------------------------------------------------------------------------------
    for epoch in range(start_epoch, config.training.scheduler.epochs + 1):
        train_metric = train(model, train_dataloader, train_data_aug, epoch, config, optimizer,
                             loss_func, train_writer, schedulers, logger, device)
        val_metric = validate(model, extra_train_dataloader, extra_train_aug, val_dataloader,
                              val_data_aug, epoch, config, val_writer, logger, device)

        if val_metric.better_than(best_metrics):
            best_metrics = val_metric
            save_checkpoint(model, optimizer, epoch, train_metric, best_metrics,
                            'pretrain-ckpt-best', args, logger)
        elif val_metric.less_than(best_metrics):
            save_checkpoint(model, optimizer, epoch, train_metric, val_metric,
                            'pretrain-ckpt-last', args, logger)
        log_string(logger, f"------------------------------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------
    # Close tensorboard writers
    # ------------------------------------------------------------------------------------------------------------------
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def train(model, train_dataloader, train_data_aug, epoch, config, optimizer, loss_func,
          train_writer, schedulers, logger, device):
    epoch_start_time = time.time()
    Avg_meter = AverageMeter(['loss_patch'])
    model.train()
    # ------------------------------------------------------------------------------------------------------------------
    # To store outputs of student and teacher models
    # ------------------------------------------------------------------------------------------------------------------
    student_outputs = []
    teacher_outputs = []
    for idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training Progress"):
        # --------------------------------------------------------------------------------------------------------------
        # Update learning rate and weight decay (no weight decay for first group)
        # --------------------------------------------------------------------------------------------------------------
        iteration = len(train_dataloader) * epoch + idx
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = schedulers['lr'][iteration]
            if i == 1:
                param_group["weight_decay"] = schedulers['wd'][iteration]
        # --------------------------------------------------------------------------------------------------------------
        # Data augmentation
        # --------------------------------------------------------------------------------------------------------------
        data = data.float().to(device)
        data = train_data_aug(data)
        # --------------------------------------------------------------------------------------------------------------
        # Forward pass
        # --------------------------------------------------------------------------------------------------------------
        optimizer.zero_grad()
        student_output, teacher_output, codebook = model(data[:, :, :3])
        # --------------------------------------------------------------------------------------------------------------
        # Save outputs of student and teacher models
        # --------------------------------------------------------------------------------------------------------------
        student_outputs.append(student_output.detach().cpu().numpy())
        teacher_outputs.append(teacher_output.detach().cpu().numpy())
        # --------------------------------------------------------------------------------------------------------------
        # Compute loss
        # --------------------------------------------------------------------------------------------------------------
        prediction = generate_assignment(student_output, codebook, schedulers['tau_S'][iteration], is_teacher=False)
        target = generate_assignment(teacher_output, codebook, schedulers['tau_T'][iteration], is_teacher=True)
        train_loss = loss_func(prediction, target)
        # --------------------------------------------------------------------------------------------------------------
        # Update student and teacher networks
        # --------------------------------------------------------------------------------------------------------------
        train_loss.backward()
        if config.training.optimizer.clip_grad:
            clip_gradients(model, config.training.optimizer.clip_grad)
        optimizer.step()
        model.teacher_encoder.update(schedulers['ema'][iteration])
        # --------------------------------------------------------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------------------------------------------------------
        Avg_meter.update(train_loss.item(), 0)
    # ------------------------------------------------------------------------------------------------------------------
    # Record metrics
    # ------------------------------------------------------------------------------------------------------------------
    epoch_end_time = time.time()
    if train_writer is not None:
        train_writer.add_scalar('loss_patch', Avg_meter.avg(0), epoch)
    # ------------------------------------------------------------------------------------------------------------------
    # Visualize student and teacher output distributions every N epochs
    # ------------------------------------------------------------------------------------------------------------------
    if epoch % config.training.visualization_interval == 0:
        visualize_distribution(student_outputs, teacher_outputs, epoch)
    # ------------------------------------------------------------------------------------------------------------------
    # Log info
    # ------------------------------------------------------------------------------------------------------------------
    log_string(logger,
               '[Training] EPOCH: %d Time = %.3f (s) Loss = %.4f ' %
               (epoch + 1, epoch_end_time - epoch_start_time, Avg_meter.avg(0)))
    return Acc_Metric(Avg_meter.avg(0))


def validate(model, extra_train_dataloader, extra_train_aug, val_dataloader,
             val_data_aug, epoch, config, val_writer, logger, device):
    epoch_start_time = time.time()
    Avg_acc = AverageMeter(['svm_train_acc', 'svm_val_acc'])
    model.eval()
    test_features = []
    test_label = []
    train_features = []
    train_label = []
    for idx, data in enumerate(tqdm(extra_train_dataloader, desc="Validation Progress")):
        points = data[0].to(device)
        label = data[1].to(device)
        points = extra_train_aug(points)
        with torch.no_grad():
            feature, _ = model.forward_test(points[:, :, :3])
            feature = torch.cat([feature.max(dim=1).values, feature.mean(dim=1)], dim=-1)
        target = label.view(-1)
        train_features.append(feature.detach())
        train_label.append(target.detach())
    for idx, data in enumerate(tqdm(val_dataloader, desc="Validation Progress")):
        points = data[0].to(device)
        label = data[1].to(device)
        points = val_data_aug(points)
        with torch.no_grad():
            feature, _ = model.forward_test(points[:, :, :3])
            feature = torch.cat([feature.max(dim=1).values, feature.mean(dim=1)], dim=-1)
        target = label.view(-1)
        test_features.append(feature.detach())
        test_label.append(target.detach())
    epoch_end_time = time.time()
    # ------------------------------------------------------------------------------------------------------------------
    # Train and evaluate SVM classifier
    # ------------------------------------------------------------------------------------------------------------------
    train_features = torch.cat(train_features, dim=0)
    train_label = torch.cat(train_label, dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_label = torch.cat(test_label, dim=0)
    # ------------------------------------------------------------------------------------------------------------------
    # Visualize TSNE distribution every N epochs
    # ------------------------------------------------------------------------------------------------------------------
    if epoch % config.training.visualization_interval == 0:
        visualize_tsne(train_features, train_label, epoch)
    # ------------------------------------------------------------------------------------------------------------------
    # Train SVM classifier
    # ------------------------------------------------------------------------------------------------------------------
    svm_acc = evaluate_svm_gpu(train_features, train_label,
                               test_features, test_label)
    Avg_acc.update(svm_acc['train_acc'], 0)
    Avg_acc.update(svm_acc['val_acc'], 1)
    # ------------------------------------------------------------------------------------------------------------------
    # Record average accuracy and print
    # ------------------------------------------------------------------------------------------------------------------
    if val_writer is not None:
        val_writer.add_scalar('svm_train_acc', svm_acc['train_acc'], epoch)
        val_writer.add_scalar('svm_val_acc', svm_acc['val_acc'], epoch)
    log_string(logger, '[Validation] EPOCH: %d Time = %.3f (s) svm_train_acc = %.4f, svm_val_acc = %.4f' %
               (epoch + 1, epoch_end_time - epoch_start_time, svm_acc['train_acc'], svm_acc['val_acc']))
    return Acc_Metric(Avg_acc.avg(1))
