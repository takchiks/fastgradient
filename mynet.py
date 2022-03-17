import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from mynet_utils import PointNetEncoder, feature_transform_reguliarzer
import numpy as np

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)


        self.fc0 = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k)
        self.dropout = nn.Dropout(p=0.2)
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.sigmoid(self.bn0(self.fc0(x))/2+1)
        x = F.sigmoid(self.bn1(self.fc1(x))/2+1)
        x = F.sigmoid(self.bn2(self.fc2(x))/2+1)
        x = F.sigmoid(self.bn3(self.fc4(x))/2+1)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
