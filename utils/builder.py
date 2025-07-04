import torch.optim as optim
from torch.utils.data import DataLoader

from utils import registry
from utils.main_util import cosine_scheduler, add_weight_decay
from utils.main_util import plot_curve

DATASETS = registry.Registry('dataset')
MODELS = registry.Registry('model')


def dataset_builder(config):
    """
    Build dataset and dataloader
    @param config: configuration for dataset and dataloader
    @return: dataloader instance
    """
    dataset = DATASETS.build(config.base, config.others)
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=config.shuffle,
                            drop_last=config.drop_last,
                            num_workers=config.num_workers)
    return dataloader


def model_builder(config):
    """
    Build model
    @param config: model configuration
    @return: model instance
    """
    model = MODELS.build(config)
    return model


def scheduler_builder(config, train_dataloader):
    """
    Build schedulers for learning rate, weight decay, EMA momentum, etc. (only for parameters present in config)
    @param train_dataloader: training dataloader, used to calculate number of batches per epoch
    @param config: scheduler configuration
    @return: dictionary of schedulers
    """
    batch_num = len(train_dataloader)
    schedulers = {}

    if config.type == 'CosSche':
        # Learning rate scheduler (required)
        schedulers['lr'] = cosine_scheduler(
            start_warmup_value=config.warmup_lr,
            base_value=config.base_lr,
            final_value=config.final_lr,
            warmup_epochs=config.lr_warmup_epochs,
            epochs=config.epochs,
            batch_num=batch_num)
        plot_curve(schedulers['lr'], "Learning Rate")

        # Weight decay scheduler (required)
        schedulers['wd'] = cosine_scheduler(
            start_warmup_value=config.warmup_weight_decay,
            base_value=config.base_weight_decay,
            final_value=config.final_weight_decay,
            warmup_epochs=config.wd_warmup_epochs,
            epochs=config.epochs,
            batch_num=batch_num)
        plot_curve(schedulers['wd'], "Weight Decay")

        # EMA momentum scheduler (optional)
        if hasattr(config, 'warmup_ema'):
            schedulers['ema'] = cosine_scheduler(
                start_warmup_value=config.warmup_ema,
                base_value=config.base_ema,
                final_value=config.final_ema,
                warmup_epochs=getattr(config, 'ema_warmup_epochs', 0),
                epochs=config.epochs,
                batch_num=batch_num)
            plot_curve(schedulers['ema'], "EMA Momentum")

        # Teacher temperature scheduler (optional)
        if hasattr(config, 'warmup_tau_T'):
            schedulers['tau_T'] = cosine_scheduler(
                start_warmup_value=config.warmup_tau_T,
                base_value=config.base_tau_T,
                final_value=config.final_tau_T,
                warmup_epochs=getattr(config, 'tau_T_warmup_epochs', 0),
                epochs=config.epochs,
                batch_num=batch_num)
            plot_curve(schedulers['tau_T'], "tau teacher")

        # Student temperature scheduler (optional)
        if hasattr(config, 'warmup_tau_S'):
            schedulers['tau_S'] = cosine_scheduler(
                start_warmup_value=config.warmup_tau_S,
                base_value=config.base_tau_S,
                final_value=config.final_tau_S,
                warmup_epochs=getattr(config, 'tau_S_warmup_epochs', 0),
                epochs=config.epochs,
                batch_num=batch_num)
            plot_curve(schedulers['tau_S'], "tau student")
    else:
        raise NotImplementedError(f"Scheduler type {config.type} is not implemented.")

    return schedulers


def optimizer_builder(config, model):
    """
    Build optimizer
    @param model: model to optimize
    @param config: optimizer configuration including type and parameters
    @return: optimizer instance
    """
    trainable_params = add_weight_decay(model)
    if config.type == 'AdamW':
        optimizer = optim.AdamW(params=trainable_params)
    elif config.type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.base_lr)
    elif config.type == 'SGD':
        optimizer = optim.SGD(model.parameters(), nesterov=True, lr=config.base_lr)
    else:
        raise NotImplementedError(f"Optimizer type {config.type} is not implemented.")
    return optimizer
