from .resnet import CBA
import torch.nn as nn


class VGG(nn.Module):
    def __init__(
        self, layer_classes, n_size, input_size=3, scale=1
    ):
        """
            n_size: one of [0,1,2,3] for vgg-11,13,16,19
        """
        super().__init__()

        n_repeats_all = {
            0: [1, 1, 2],
            1: [2, 2, 2],
            2: [2, 2, 3],
            3: [2, 2, 4],
        }

        if n_size not in n_repeats_all:
            print(f"Unsupported value n_size ({n_size}), can only be: 0,1,2,3")
            exit(-2)
        n_repeats = n_repeats_all[n_size]

        blocks = []

        prev_bw = input_size
        bw = int(scale*64)

        for i, n_blocks in enumerate(n_repeats):
            for _ in range(n_blocks):
                blocks.append(
                    CBA(layer_classes, prev_bw, bw, 3)
                )
                prev_bw = bw
            blocks.append(
                nn.MaxPool2d(2, 2)
            )
            if i < len(n_repeats)-2:
                bw *= 2
        self.blocks = nn.Sequential(*blocks)
        self.output_size = prev_bw

    def forward(self, nx):
        nx = self.blocks(nx)
        return nx
