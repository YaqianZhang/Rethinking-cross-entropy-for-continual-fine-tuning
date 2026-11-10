# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .simple import SimpleHead


class DynamicSimpleHead(SimpleHead):
    def __init__(
        self,
        num_classes: int = 0,
        num_features: int = 2048,
        bias: bool = True,
        pool: nn.Module = nn.AdaptiveAvgPool2d(1),
        neck: nn.Module = nn.Identity(),
        weight_initialization_type: str = "default",
        weight_std: float = 0.001,
        weight_seed: int = -1 ,
    ):
        super().__init__(num_classes, num_features, bias, pool, neck,
                         weight_initialization_type,
                         weight_std,
                         weight_seed)

    @property
    def embeddings(self):
        weights = [classifier.weight for classifier in self.classifiers]
        return torch.cat(weights)

    def setup(
        self,
        pool: nn.Module = nn.AdaptiveAvgPool2d(1),
        neck: nn.Module = nn.Identity(),
    ):
        self.pool, self.neck = pool, neck

        self.classifiers = nn.ModuleList()

        if self.num_classes > 0:
            self.append(self.num_classes)

    def append(self, num_classes: int):

        args = [self.num_features, num_classes, self.bias, self.weight_initialization_type, self.weight_std,self.weight_seed]
        classifier = self._create_classifier(*args)

        self.classifiers.append(classifier)
        if len(self.classifiers) > 1 or self.num_classes == 0:
            self.num_classes += num_classes

    def classify(self, input: torch.Tensor):
        output = [classifier(input) for classifier in self.classifiers]
        output = torch.cat(output, dim=1)
        return output

    def forward(self, input):
        if self.feature_mode:
            return input
        output = self.pool(input).flatten(1)
        #output= input[:, 0]
        output = self.neck(output)

        if self.feature_mode:
            return output

        output = self.classify(output)

        return output
