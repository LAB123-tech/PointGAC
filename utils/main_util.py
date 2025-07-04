import os
import random

import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn.functional as F
from cuml.svm import LinearSVC
from thop import profile, clever_format
from torch.autograd import Variable

from utils.logger import log_string

NORMALIZE_EPS = 1e-5


def set_random_seed(seed):
    """
    set random seed
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False


def cosine_scheduler(start_warmup_value, base_value, final_value, warmup_epochs, epochs, batch_num):
    """
    Cosine scheduler with warmup:
    Starts from start_warmup_value, linearly warms up to base_value,
    then decays to final_value using a cosine schedule.

    @param start_warmup_value: Initial value at warmup start
    @param base_value: Value after warmup
    @param final_value: Final value after cosine decay
    @param warmup_epochs: Number of warmup epochs
    @param epochs: Total number of epochs
    @param batch_num: Number of batches per epoch
    :return: Numpy array containing the learning rate schedule per batch
    """
    warmup_schedule = numpy.array([])
    warmup_iters = warmup_epochs * batch_num
    # Warmup phase: linearly increase from start_warmup_value to base_value over warmup_iters steps
    if warmup_epochs > 0:
        warmup_schedule = numpy.linspace(start_warmup_value, base_value, warmup_iters)
    # Cosine decay phase: decay from base_value to final_value over remaining steps
    cos_iters = numpy.arange(epochs * batch_num - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + numpy.cos(numpy.pi * cos_iters / len(cos_iters)))
    # Concatenate warmup and cosine decay schedules
    schedule = numpy.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * batch_num
    return schedule


def add_weight_decay(model):
    """
    Do not apply weight decay to teacher encoder, tokenizer, and bias parameters.
    In EMA (Exponential Moving Average), teacher model parameters are updated from student model parameters,
    so teacher parameters don't need weight decay.
    Tokenizer parameters or task-specific prediction heads should not have weight decay,
    as it may limit their expressive ability and hurt performance.
    Bias terms adjust model output flexibility; applying weight decay to them
    may reduce this flexibility and impair fitting ability.

    @param model: model with parameters
    return: list of dicts separating params with and without weight decay
    """
    names_no_grad = []
    names_no_decay = []
    names_decay = []
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        # Skip params that do not require gradient (e.g. teacher encoder)
        if not param.requires_grad:
            names_no_grad.append(name)
            continue
        # No weight decay for tokenizer params, bias params, or 1D params
        if 'token' in name or name.endswith(".bias") or len(param.shape) == 1:
            names_no_decay.append(name)
            no_decay.append(param)
        else:
            names_decay.append(name)
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay}]


def create_experiment_dir(args):
    """
    create directory
    @param args:
    @return:
    """
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
        print('Create log path successfully at %s' % args.log_path)
    if not os.path.exists(args.save_ckpts):
        os.makedirs(args.save_ckpts)
        print('Create ckpt path successfully at %s' % args.save_ckpts)


def cal_params(base_model, device):
    """
    Calculate FLOPs and the number of parameters required for model training.
    @param base_model:
    @param device:
    return:
    """
    flops, params = profile(base_model, (Variable(torch.rand(1, 1024, 3)).to(device), False))
    flops, params = clever_format([flops, params], '%.3f')
    print(f"Computational cost：{flops}, Number of parameters：{params}")
    param_size = sum(p.numel() for p in base_model.parameters()) * 4
    param_size_mb = param_size / (1024 ** 2)
    print(f"model size：{param_size_mb:.3f} MB")


def load_model(args, logger, is_train, **kwargs):
    """
    Load a model for training or testing.
    @param args: Argument settings
    @param logger: Logger instance
    @param is_train: Boolean flag, True for resuming training, False for testing
    @param kwargs: Other parameters such as student model, teacher model, optimizer, etc.
    @return: If resuming training, returns the starting epoch and best performance metrics; otherwise returns None
    """
    ckpt_path = os.path.join(args.save_ckpts, 'pretrain-ckpt-best.pth')
    if not os.path.exists(ckpt_path):
        log_string(logger, f'[ERROR] Checkpoint file {ckpt_path} does not exist!')
        raise FileNotFoundError(f'Checkpoint file {ckpt_path} does not exist!')
    # ------------------------------------------------------------------------------------------------------------------
    # Load model parameters
    # ------------------------------------------------------------------------------------------------------------------
    map_location = f'cuda:{args.device}' if args.device != 'cpu' else 'cpu'
    state_dict_gpu = torch.load(ckpt_path, map_location=map_location)
    if is_train:
        # In training mode, load student model, teacher model, optimizer and training state
        kwargs["model"].load_state_dict(state_dict_gpu["model"], strict=True)
        kwargs["optimizer"].load_state_dict(state_dict_gpu["optimizer"])
        # --------------------------------------------------------------------------------------------------------------
        # Visualize the update frequency of dictionary vectors
        # --------------------------------------------------------------------------------------------------------------
        draw_heatmap(kwargs["model"].codebook_gen.usage_history)
        start_epoch = state_dict_gpu.get('epoch', 0) + 1
        best_metrics = state_dict_gpu.get('best_metrics', {}).get('acc', 0.0)
        log_string(logger, f'[RESUME] Resuming checkpoint at epoch {start_epoch - 1} (best_metrics = {best_metrics})')
        return start_epoch, best_metrics
    else:
        # In testing mode, load only the student model
        kwargs["model"].load_state_dict(state_dict_gpu["model"], strict=False)
        log_string(logger, f'[LOAD] Loaded checkpoint from {ckpt_path}')
        return None


def load_model_to_finetune(base_model, logger):
    """
    Load pretrained weights to start fine-tuning training.
    @param base_model: The model to load weights into
    @param logger: Logger instance
    @return: None
    """
    # Load pretrained checkpoint
    ckpt = torch.load("checkpoints/pretrain/pretrain-ckpt-best.pth")
    base_ckpt = {k: v for k, v in ckpt['model'].items()}
    # ------------------------------------------------------------------------------------------------------------------
    # Rename parameter keys: change prefix 'student_model.encoder' to 'model'
    # Because downstream task encoder names start with 'model', while pretrained model keys start with 'student_model.encoder'
    # ------------------------------------------------------------------------------------------------------------------
    for k in list(base_ckpt.keys()):
        if k.startswith('student_model.encoder'):
            new_key = 'model' + k[len('student_model.encoder'):]
            base_ckpt[new_key] = base_ckpt[k]
            del base_ckpt[k]
    # ------------------------------------------------------------------------------------------------------------------
    # Load parameters and check for compatibility
    # ------------------------------------------------------------------------------------------------------------------
    incompatible = base_model.load_state_dict(base_ckpt, strict=False)

    # Calculate counts of keys
    model_keys = set(base_model.state_dict().keys())
    loaded_keys = set(base_ckpt.keys())

    # Successfully loaded parameters = model parameters ∩ checkpoint parameters
    successfully_loaded = model_keys & loaded_keys

    # Log summary
    log_string(logger, '[Pretrain] Successfully loaded checkpoint from pretrain-ckpt-best.pth')
    log_string(logger, f'[Pretrain] Total parameters in model: {len(model_keys)}')
    log_string(logger, f'[Pretrain] Total parameters in checkpoint: {len(loaded_keys)}')
    log_string(logger, f'[Pretrain] Successfully loaded parameters: {len(successfully_loaded)}')
    log_string(logger, f'[Pretrain] Missing parameters: {len(incompatible.missing_keys)}')
    log_string(logger, f'[Pretrain] Unexpected parameters: {len(incompatible.unexpected_keys)}')

    # Detailed log for missing keys (up to 10)
    if incompatible.missing_keys:
        log_string(logger, '\n[Pretrain] Missing keys details:')
        for i, key in enumerate(incompatible.missing_keys[:10]):
            log_string(logger, f'{i + 1}. {key}')
        if len(incompatible.missing_keys) > 10:
            log_string(logger, f'... and {len(incompatible.missing_keys) - 10} more')

    # Detailed log for unexpected keys (up to 10)
    if incompatible.unexpected_keys:
        log_string(logger, '\n[Pretrain] Unexpected keys details:')
        for i, key in enumerate(incompatible.unexpected_keys[:10]):
            log_string(logger, f'{i + 1}. {key}')
        if len(incompatible.unexpected_keys) > 10:
            log_string(logger, f'... and {len(incompatible.unexpected_keys) - 10} more')


def save_checkpoint(model, optimizer, epoch, train_metric, best_metrics, prefix, args,
                    logger=None):
    """
    save trained checkpoints
    :param model:
    :param optimizer:
    :param epoch:
    :param train_metric:
    :param best_metrics:
    :param prefix:
    :param args:
    :param logger:
    :return:
    """
    path = str(os.path.join(args.save_ckpts, prefix + '.pth'))
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    if hasattr(train_metric, 'state_dict'):
        checkpoint['train_metrics'] = train_metric.state_dict()
    if hasattr(best_metrics, 'state_dict'):
        checkpoint['best_metrics'] = best_metrics.state_dict()
    torch.save(checkpoint, path)
    log_string(logger, f"[Save] Save checkpoint at {os.path.join(args.save_ckpts, prefix + '.pth')}")


def cls2onehot(y, num_classes, device_main):
    """
    Convert labels to one-hot encoding
    :param y:
    :param num_classes:
    :param device_main:
    :return:
    """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.to(device_main)
    return new_y


def clip_gradients(model, clip):
    """
    Gradient clipping
    :param model:
    :param clip:
    :return:
    """
    names_clip = []
    names_no_clip = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
            names_clip.append(name)
        else:
            names_no_clip.append(name)


def evaluate_svm_gpu(train_features, train_labels, test_features, test_labels):
    """
    Train a support vector machine classifier using cuml's LinearSVC and evaluate its classification accuracy on the test set.
    :param train_features: Training set features
    :param train_labels: Training set labels
    :param test_features: Test set features
    :param test_labels: Test set labels
    :return: Dictionary containing classification accuracy for both training and test sets
    """
    svm = LinearSVC(C=0.012)
    svm.fit(train_features, train_labels)
    train_acc = svm.score(train_features, train_labels)
    val_acc = svm.score(test_features, test_labels)
    return {'train_acc': train_acc, 'val_acc': val_acc}


def plot_curve(data, curve_str):
    """
    Visualize the curve of learning rate, temperature, or weight decay.
    @param data: The data points to plot
    @param curve_str: The name or type of the curve
    @return: None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Data Curve')
    plt.title('Curve Plot of the {}'.format(curve_str))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/curve/{curve_str}.pdf")
    plt.close()  # Release plotting resources


def draw_heatmap(data, rows=16, cols=128, cmap='viridis', interpolation='nearest', title='Update Count Heatmap'):
    """
    Draw a heatmap.
    @param data: input data (1D tensor or array)
    @param rows: number of rows in the heatmap
    @param cols: number of columns in the heatmap
    @param cmap: colormap used for visualization
    @param interpolation: interpolation method for imshow
    @param title: title of the heatmap
    @return: None
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check that input length matches rows * cols
    # ------------------------------------------------------------------------------------------------------------------
    assert data.shape[0] == rows * cols, f"Input data length must be {rows * cols}, but got {data.size}."
    # ------------------------------------------------------------------------------------------------------------------
    # Normalize data to [0, 1] and reshape to 2D matrix
    # ------------------------------------------------------------------------------------------------------------------
    data = (data - data.min()) / (data.max() - data.min())
    matrix = data.reshape(rows, cols).detach().cpu().numpy()
    # ------------------------------------------------------------------------------------------------------------------
    # Draw heatmap
    # ------------------------------------------------------------------------------------------------------------------
    plt.imshow(matrix, cmap=cmap, interpolation=interpolation)
    plt.colorbar(label='Update Count')
    plt.xlabel('Dictionary Index (Column)')
    plt.ylabel('Dictionary Index (Row)')
    plt.title(title)
    plt.grid(False)
    plt.savefig(f"output/curve/Dic_usage.pdf", format='pdf')
    plt.show()
    plt.close()  # Release plotting resources


def generate_assignment(feature, dictionary, tau, is_teacher=False):
    """
    Compute the distribution (assignment) of teacher features over the dictionary.
    @param feature: input feature tensor
    @param dictionary: dictionary tensor
    @param tau: temperature parameter
    @param is_teacher: if True, use softmax; if False, use log_softmax for stability
    @return: assignment probabilities or log probabilities
    """
    normed_feature = F.normalize(feature, p=2, dim=-1, eps=NORMALIZE_EPS)
    normed_dictionary = F.normalize(dictionary, p=2, dim=-1, eps=NORMALIZE_EPS)
    similarity = torch.matmul(normed_feature, normed_dictionary.t())
    # ------------------------------------------------------------------------------------------------------------------
    # For loss calculation (KL divergence), student's assignment needs log_softmax to ensure numerical stability.
    # Using log_softmax is more stable than applying softmax then log separately.
    # ------------------------------------------------------------------------------------------------------------------
    if is_teacher:
        assignment = torch.softmax(similarity / tau, dim=1)
    else:
        assignment = torch.log_softmax(similarity / tau, dim=-1)
    return assignment
