import torch.nn as nn


class SmoothL1_Loss_Func:
    def __init__(self, beta=2):
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=beta)

    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        """
        Compute the loss between prediction and target.
        @param pred: Predictions, shape (num_token, num_classes)
        @param target: Targets, shape (num_token, num_classes)
        @return: Smooth L1 loss
        """
        smooth_l1_loss = self.smooth_l1_loss(pred, target)
        return smooth_l1_loss


class KL_Loss_Func:
    def __init__(self):
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        """
        Compute the KL divergence loss.
        @param pred: Predictions, shape (10, 8192), already log_softmaxed
        @param target: Targets, shape (10, 8192), already softmaxed
        @return: KL divergence loss
        """
        return self.kl_loss(pred, target)


class CrossEntropyLoss_Func:
    def __init__(self):
        self.ce_loss = nn.CrossEntropyLoss()

    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        """
        Compute cross-entropy loss.
        @param pred: Predictions, shape (batch_size, num_classes), raw logits (no softmax needed)
        @param target: Targets, shape (batch_size,) for class indices, or (batch_size, num_classes) for class probabilities
        @return: Cross-entropy loss
        """
        return self.ce_loss(pred, target)
