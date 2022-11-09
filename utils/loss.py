import torch.nn as nn


def get_criterion(TRG_PAD_IDX):
    return nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)