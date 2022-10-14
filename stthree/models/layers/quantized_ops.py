import torch
import torch.nn.functional as F


# -- Binary quantization
# Weights
def binarize(xx):
    return xx.sign()


# Activations
def binarize_tanh(xx):
    xx = F.hardtanh(xx, min_val=-1, max_val=1)
    with torch.no_grad():
        xx = xx.sign_()
    return xx


def binarize_relu(xx, rounding=True):
    xx = F.hardtanh(xx, min_val=0, max_val=1)
    with torch.no_grad():
        if rounding:
            xx = xx.round_()
        else:
            xx = xx.ceil_()
    return xx


def binarize_relu_strict(xx):
    return binarize_relu(xx, rounding=False)


# -- Ternary quantization
# Weights
def _ternarize(xx, cutoff):
    zeros = torch.zeros(xx.shape, device=xx.device)
    ones = torch.ones(xx.shape, device=xx.device)

    return torch.where(xx > cutoff, ones,
                       torch.where(xx <= -cutoff, -ones, zeros))


def ternarize(xx):
    cutoff = (0.7 * xx.abs().mean())
    return _ternarize(xx, cutoff)


def ternarize_simple(xx):
    return _ternarize(xx, 0.5)


# Activations
def ternarize_tanh(xx):
    xx = F.hardtanh(xx, min_val=-1, max_val=1)
    with torch.no_grad():
        xx = xx.round_()
    return xx


def ternarize_relu(xx, rounding=True):
    xx = F.hardtanh(xx, min_val=0, max_val=1)
    with torch.no_grad():
        if rounding:
            xx = xx.mul_(2).round_().div_(2)
        else:
            xx = xx.mul_(2).ceil_().div_(2)
    return xx


def ternarize_relu_strict(xx):
    return ternarize_relu(xx, rounding=False)


# -- Q-bit quantization
# Weights
def quantize(xx, nbits=4):
    non_sign_bits = nbits-1
    limit = 2**non_sign_bits
    max_val = 1-1/limit
    xx = F.hardtanh(xx, min_val=-1, max_val=max_val)
    xx = xx.mul_(limit).round_().div_(limit)
    return xx


def quantize_16(xx):
    return quantize(xx, nbits=16)


def quantize_8(xx):
    return quantize(xx, nbits=8)


def quantize_2(xx):
    return quantize(xx, nbits=2)


# Activations
def quantize_tanh(xx, nbits=4):
    non_sign_bits = nbits-1
    limit = 2**non_sign_bits
    max_val = 1-1/limit
    xx = F.hardtanh(xx, min_val=-1, max_val=max_val)
    with torch.no_grad():
        xx = xx.mul_(limit).round_().div_(limit)
    return xx


def quantize_relu(xx, nbits=4):
    limit = 2**nbits
    max_val = 1-1/limit
    xx = F.hardtanh(xx, min_val=0, max_val=max_val)
    with torch.no_grad():
        xx = xx.mul_(limit).round_().div_(limit)
    return xx


def quantize_relu_16(xx):
    return quantize_relu(xx, nbits=16)


def quantize_relu_8(xx):
    return quantize_relu(xx, nbits=8)


def quantize_relu_2(xx):
    return quantize_relu(xx, nbits=2)


def quantize_tanh_16(xx):
    return quantize_tanh(xx, nbits=16)


def quantize_tanh_8(xx):
    return quantize_tanh(xx, nbits=8)


def quantize_tanh_2(xx):
    return quantize_tanh(xx, nbits=2)


# -- Floating point quantization
def hard_tanh(xx):
    return F.hardtanh(xx)


def hard_relu(xx):
    return F.hardtanh(xx, min_val=0)
