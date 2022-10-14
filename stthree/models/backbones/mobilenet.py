import torch.nn as nn
from .resnet import CBA

class MobileNetv1(nn.Module):
    def __init__(self, layer_classes, input_size):
        super().__init__()

        def conv_bn(inp, oup, stride):
            return CBA(layer_classes, inp, oup, 3, stride=stride, bias=False)

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                CBA(layer_classes, inp, inp, 3, stride=stride, depthwise=True, bias=False),
                # pw
                CBA(layer_classes, inp, oup, 1, bias=False)
            )

        self.model = nn.Sequential(
            conv_bn(input_size, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.output_size = 1024
    def forward(self, x):
        x = self.model(x)
        return x
