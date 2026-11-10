# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import math
class SimpleHead(nn.Module):
    feature_mode: bool = False

    def __init__(
        self,
        num_classes: int,
        num_features: int = 2048,
        bias: bool = True,
        pool: nn.Module = nn.AdaptiveAvgPool2d(1),
        neck: nn.Module = nn.Identity(),
        weight_initialization_type: str = "default",
        weight_std: float = 0.001,
        weight_seed: int = -1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.bias = bias

        self.weight_initialization_type = weight_initialization_type
        self.weight_std = weight_std
        self.weight_seed = weight_seed
        self.setup(pool, neck,) 

    @property
    def embeddings(self):
        return self.classifier.weight

    def setup(
        self,
        pool: nn.Module = nn.AdaptiveAvgPool2d(1),
        neck: nn.Module = nn.Identity(),
    ):
        self.pool, self.neck = pool, neck

        # self.weight_std = weight_std
        # self.weight_seed = weight_seed
        args = [self.num_features, self.num_classes, self.bias, self.weight_initialization_type, self.weight_std]
        self.classifier = self._create_classifier(*args)

    def classify(self, input: torch.Tensor):
        return self.classifier(input)

    def forward(self, input):
        if self.feature_mode:
            return input
        output = self.pool(input).flatten(1)
        output = self.neck(output)

        if self.feature_mode:
            return output

        output = self.classify(output)
        return output

    @staticmethod
    def weights_init(m,weight_std=0.001,weight_seed=-1):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            # nn.init.xavier_uniform_(m.weight)
            if weight_seed != -1:
                with torch.random.fork_rng():
                    torch.manual_seed(weight_seed)
                    nn.init.normal_(m.weight, std=weight_std)
            else:
                
                nn.init.normal_(m.weight, std=weight_std)

            ## original initialization
            #torch.manual_seed(40)
            #nn.init.normal_(m.weight, std=0.01)

            #nn.init.kaiming_normal_(m.weight,)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    @staticmethod
    def weights_init_bias(m ):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:

            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    def _create_classifier(
        self,
        num_features: int,
        num_classes: int,
        bias=True,
        weight_initialization_type="default",
        weight_std=0,
        weight_seed=-1,
    ):
        assert num_features > 0 and num_classes > 0

        classifier = nn.Linear(num_features, num_classes, bias=bias)
        if weight_initialization_type=="default":
            pass
        elif weight_initialization_type=="normal":

            self.weights_init(classifier,weight_std,weight_seed)
        elif weight_initialization_type=="default_bias_zero":

            self.weights_init_bias(classifier)
        else:
            raise NotImplementedError()

        return classifier
