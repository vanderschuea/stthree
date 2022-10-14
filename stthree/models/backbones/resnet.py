from turtle import forward
import torch.nn as nn
import torch.nn.functional as F


class CBA(nn.Module):
    def __init__(
        self, layer_classes, in_width, width, ksize, conv=True, bn=True, act=True, stride=1,
        depthwise=False, bias=True
    ):
        super().__init__()
        assert not depthwise or in_width == width,\
            f"DWise conv needs equal width for in-and outputs,\n" +\
            f"but got {in_width} and {width}"

        if conv:
            padding = ksize // 2
            self.conv = layer_classes.conv(
                in_width, width, ksize, stride=stride, padding=padding,
                padding_mode='zeros', groups=(in_width if depthwise else 1),
                bias=bias
            )
        if bn:
            self.batchnorm = layer_classes.bn(width, momentum=0.01)
        if act:
            self.act = layer_classes.act()

    def forward(self, nx):
        if hasattr(self, "conv"):
            nx = self.conv(nx)
        if hasattr(self, "batchnorm"):
            nx = self.batchnorm(nx)
        if hasattr(self, "act"):
            nx = self.act(nx)
        return nx


class ResBlock(nn.Module):
    def __init__(self, layer_classes, in_width, width, ksize=3, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            CBA(layer_classes, in_width, width, ksize, stride=stride),
            CBA(layer_classes, width,    width, ksize, act=False),
        )
        self.skip = nn.Sequential() if in_width == width and stride == 1 else \
            CBA(layer_classes, in_width, width, 1, stride=stride, act=False)
        self.last_act = layer_classes.act()

    def forward(self, nx):
        main = self.block(nx)
        shortcut = self.skip(nx)
        nx = main + shortcut
        return self.last_act(nx)


class ResNet(nn.Module):
    def __init__(self, layer_classes, n_size, input_size, scale=1):
        """
        """
        super().__init__()

        res_blocks = []
        bw = int(16*scale)
        prev_bw = new_bw = bw
        self.start_block = CBA(layer_classes, input_size, new_bw, 3)
        for i in range(3):
            for j in range(n_size):
                stride = 1 if i == 0 or j > 0 else 2
                res_blocks.append(
                    ResBlock(layer_classes, prev_bw, new_bw, 3, stride=stride)
                )
                prev_bw = new_bw
            bw *= 2
            new_bw = bw
        self.res_blocks = nn.Sequential(*res_blocks)
        self.output_size = prev_bw

    def forward(self, nx):
        nx = self.start_block(nx)
        nx = self.res_blocks(nx)
        return nx


class ResBlockv2(nn.Module):
    def __init__(self, layer_classes, in_width, width, ksize=3, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            CBA(layer_classes, None, in_width, None, conv=False),
            CBA(layer_classes, in_width, width, ksize, stride=stride),
            CBA(layer_classes, width,    width, ksize, act=False, bn=False),
        )
        self.skip = (lambda xx: xx) if in_width == width and stride == 1 else \
            (lambda xx: F.pad(xx[:, :, ::2, ::2], (0, 0, 0, 0, width//4, width//4), "constant", 0))

    def forward(self, nx):
        main = self.block(nx)
        shortcut = self.skip(nx)
        nx = main + shortcut
        return nx


class ResNetv2(nn.Module):
    def __init__(self, layer_classes, n_size, input_size):
        """
        """
        super().__init__()

        res_blocks = []
        bw = 16
        prev_bw = new_bw = bw
        self.start_block = CBA(layer_classes, input_size, new_bw, 3, act=False)
        for i in range(3):
            for j in range(n_size):
                stride = 1 if i == 0 or j > 0 else 2
                res_blocks.append(
                    ResBlockv2(layer_classes, prev_bw, new_bw, 3, stride=stride)
                )
                prev_bw = new_bw
            bw *= 2
            new_bw = bw
        self.res_blocks = nn.Sequential(*res_blocks)
        self.output_size = prev_bw

    def forward(self, nx):
        nx = self.start_block(nx)
        nx = self.res_blocks(nx)
        return nx


class BottleneckResBlock(nn.Module):
    def __init__(self, layer_classes, in_width, width, expansion, ksize=3, stride=1):
        super().__init__()
        out_width = width*expansion
        self.block = nn.Sequential(
            CBA(layer_classes, in_width, width,     1,     bias=False),
            CBA(layer_classes, width,    width,     ksize, stride=stride, bias=False),
            CBA(layer_classes, width,    out_width, 1,     bias=False, act=False),
        )
        self.skip = nn.Sequential() if in_width == out_width and stride == 1 else \
            CBA(layer_classes, in_width, out_width, 1, stride=stride, act=False, bias=False)
        self.last_act = layer_classes.act()

    def forward(self, nx):
        main = self.block(nx)
        shortcut = self.skip(nx)
        nx = main + shortcut
        return self.last_act(nx)

class BigResNet(nn.Module):
    def __init__(self, layer_classes, n_size, input_size, init_stride=2):
        assert n_size in {18,34,50,101,152}, f"n_size of {n_size} is not supported"
        super().__init__()
        StructBlock = ResBlock if n_size < 50 else BottleneckResBlock
        repetitions = {
            18:[2, 2, 2, 2],
            34:[3,4,6,3],
            50:[3,4,6,3],
            101:[3,4,23,3],
            152:[3,8,36,3],
        }

        prev_width = width = 64
        expansion = 4
        self.start_block = CBA(layer_classes, input_size, width, 7, bias=False, stride=init_stride)

        res_blocks = []
        width = 64
        for irepet, repetition in enumerate(repetitions[n_size]):
            block = []
            for j in range(repetition):
                stride = 2 if j==0 and irepet > 0 else 1
                if StructBlock is ResBlock:
                    block.append(StructBlock(layer_classes, prev_width, width, stride=stride))
                    prev_width = width
                elif StructBlock is BottleneckResBlock:
                    block.append(StructBlock(layer_classes, prev_width, width, stride=stride, expansion=expansion))
                    prev_width = width*expansion
            res_blocks.append(nn.Sequential(*block))
            width *= 2

        self.block1 = res_blocks[0]
        self.block2 = res_blocks[1]
        self.block3 = res_blocks[2]
        self.block4 = res_blocks[3]

        self.output_size = prev_width

    def forward(self, nx):
        nx = self.start_block(nx)
        nx = F.max_pool2d(nx, kernel_size=3, stride=2, padding=1)
        nx = self.block1(nx)
        nx = self.block2(nx)
        nx = self.block3(nx)
        nx = self.block4(nx)
        return nx


class WideBlock(nn.Module):
    def __init__(self, layer_classes, in_width, width, ksize=3, stride=1, dropout=0.0):
        super().__init__()

        self.block = nn.Sequential(
            CBA(layer_classes, None, in_width, None, conv=False),
            CBA(layer_classes, in_width, width, ksize, stride=stride),
            nn.Dropout2d(dropout),
            CBA(layer_classes, width,    width, ksize, act=False, bn=False),
        )
        self.skip = (lambda xx: xx) if in_width == width and stride == 1 else \
            CBA(layer_classes, in_width, width, 1, stride=stride, bn=False, act=False)

    def forward(self, nx):
        main = self.block(nx)
        shortcut = self.skip(nx)
        nx = main + shortcut
        return nx


class WideResNet(nn.Module):
    def __init__(self, layer_classes, n_size, input_size, scale, dropout):
        assert ((n_size-4)%6 ==0), 'WideResNet n_size should be 6n+4'
        super().__init__()
        res_blocks = []
        prev_bw = new_bw = 16
        self.start_block = CBA(layer_classes, input_size, new_bw, 3, act=False, bn=False)
        new_bw = bw = int(16*scale)
        depth = int((n_size-4)/6)
        for i in range(3):
            for j in range(depth):
                stride = 1 if i == 0 or j > 0 else 2
                res_blocks.append(
                    WideBlock(layer_classes, prev_bw, new_bw, 3, stride=stride, dropout=dropout)
                )
                prev_bw = new_bw

            bw *= 2
            new_bw = bw
        self.res_blocks = nn.Sequential(*res_blocks)
        self.last_act = CBA(layer_classes, None, prev_bw, None, conv=False)
        self.output_size = prev_bw

    def forward(self, nx):
        nx = self.start_block(nx)
        nx = self.res_blocks(nx)
        nx = self.last_act(nx)
        return nx