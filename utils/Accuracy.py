import torch

seg_classes = {'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11],
               'Chair': [12, 13, 14, 15], 'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21],
               'Knife': [22, 23], 'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29],
               'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40],
               'Rocket': [41, 42, 43], 'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]}
seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def cal_pred_choice(seg_pred, seg_label):
    """
    Calculate the predicted classes for a batch of data
    @param seg_pred: tensor, (B, N, 50)
    @param seg_label: tensor, (B, N)
    @return: pred_choice: tensor, (B, N)
    """
    batch_size, npoint = seg_pred.shape[0], seg_pred.shape[1]
    pred_choice = torch.zeros((batch_size, npoint), dtype=torch.int64, device=seg_pred.device)
    for i in range(batch_size):
        single_class_name = seg_label_to_cat[seg_label[i, 0].item()]
        single_pred_val = seg_pred[i, :, :]
        single_cls_channel_num = seg_classes[single_class_name]
        # 获取指定通道的预测值
        relevant_pred = single_pred_val[:, single_cls_channel_num]
        # 计算预测的类别
        pred_choice[i, :] = torch.argmax(relevant_pred, dim=1) + single_cls_channel_num[0]
    return pred_choice


def cal_acc(pred_choice, seg_label, acc):
    """
    Calculate the accuracy of a batch of data
    @param pred_choice: tensor, (B, N)
    @param seg_label: tensor, (B, N)
    @param acc: []
    @return:
    """
    batch_size, npoint = pred_choice.shape[0], pred_choice.shape[1]
    pred_choice = pred_choice.view(-1, 1)
    seg_label = seg_label.view(-1, 1)
    correct = pred_choice.eq(seg_label.data).sum()
    acc.update(correct.item() / (batch_size * npoint), 0)
    return acc


def cal_ins_cls_iou(pred_choice, seg_label, Avg_acc_ins_iou, mean_cls_iou):
    """
    Calculate the instance IoU and class IoU for a batch of data
    @param pred_choice: tensor, (B, N)
    @param seg_label: tensor, (B, N)
    @param Avg_acc_ins_iou: []
    @param mean_cls_iou: {'Airplane': [], ..., 'Table': []}
    @return:
    """
    batch_size, npoint = pred_choice.shape[0], pred_choice.shape[1]
    for i in range(batch_size):
        single_pred_choice = pred_choice[i, :]
        single_seg_label = seg_label[i, :]
        single_class_name = seg_label_to_cat[single_seg_label[0].item()]
        single_part_iou = torch.zeros(len(seg_classes[single_class_name]), device=pred_choice.device)
        for idx, part_num in enumerate(seg_classes[single_class_name]):
            pred_is_part = (single_pred_choice == part_num).bool()
            label_is_part = (single_seg_label == part_num).bool()
            pred_and_label_empty = (torch.sum(label_is_part).item() == 0) and (torch.sum(pred_is_part).item() == 0)
            if pred_and_label_empty:
                single_part_iou[idx] = 1.0
            else:
                iou_up = torch.sum(label_is_part & pred_is_part).float()
                iou_down = torch.sum(label_is_part | pred_is_part).float()
                single_part_iou[idx] = iou_up / iou_down
        mean_iou = torch.mean(single_part_iou).item()
        Avg_acc_ins_iou.update(mean_iou, 1)
        mean_cls_iou[single_class_name].append(mean_iou)
    return Avg_acc_ins_iou, mean_cls_iou
