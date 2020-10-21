#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Xingchen Song @ 2020-10-13

import torch.nn as nn


class DenseModel(nn.Module):

    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.dense_layers = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(True), nn.Dropout(p=0.1),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(True), nn.Dropout(p=0.1)
        )
        self.output_layers = nn.Sequential(
            nn.Linear(hidden, out_dim), nn.LogSoftmax(dim=-1), nn.Dropout(p=0.)
        )

    def forward(self, x):
        x = self.dense_layers(x)
        x = self.output_layers(x)
        return x
