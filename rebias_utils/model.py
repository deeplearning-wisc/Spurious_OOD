"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Implementation for simple statcked convolutional networks.
"""
import torch
import torch.nn as nn


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=None, kernel_size=7, feature_pos='post'):
        super(SimpleConvNet, self).__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        self.extracter = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if feature_pos not in ['pre', 'post', 'logits']:
            raise ValueError(feature_pos)

        self.feature_pos = feature_pos

    def forward(self, x, logits_only=False):
        pre_gap_feats = self.extracter(x)
        post_gap_feats = self.avgpool(pre_gap_feats)
        post_gap_feats = torch.flatten(post_gap_feats, 1)
        logits = self.fc(post_gap_feats)

        if logits_only:
            return logits

        elif self.feature_pos == 'pre':
            feats = pre_gap_feats
        elif self.feature_pos == 'post':
            feats = post_gap_feats
        else:
            feats = logits
        return logits, feats

class ReBiasModels(object):
    """A container for the target network and the intentionally biased network.
    """
    def __init__(self, f_net, g_nets):
        self.f_net = f_net
        self.g_nets = g_nets

    def to(self, device):
        self.f_net.to(device)
        for g_net in self.g_nets:
            g_net.to(device)

    def to_parallel(self, device):
        self.f_net = nn.DataParallel(self.f_net.to(device))
        for i, g_net in enumerate(self.g_nets):
            self.g_nets[i] = nn.DataParallel(g_net.to(device))

    def load_models(self, state_dict):
        self.f_net.load_state_dict(state_dict['f_net'])
        for g_net, _state_dict in zip(self.g_nets, state_dict['g_nets']):
            g_net.load_state_dict(_state_dict)

    def train_f(self):
        self.f_net.train()

    def eval_f(self):
        self.f_net.eval()

    def train_g(self):
        for g_net in self.g_nets:
            g_net.train()

    def eval_g(self):
        for g_net in self.g_nets:
            g_net.eval()

    def train(self):
        self.train_f()
        self.train_g()

    def eval(self):
        self.eval_f()
        self.eval_g()

    def forward(self, x):
        f_pred, f_feat = self.f_net(x)
        g_preds, g_feats = [], []
        for g_net in self.g_nets:
            _g_pred, _g_feat = g_net(x)
            g_preds.append(_g_pred)
            g_feats.append(_g_feat)

        return f_pred, g_preds, f_feat, g_feats

    def __call__(self, x):
        return self.forward(x)