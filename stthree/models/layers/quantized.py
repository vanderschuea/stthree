import torch
import torch.nn as nn
import torch.nn.functional as F
from . import quantized_ops


""" Quantization Layers for N-bit quantization
"""

def str2layer(qtype):
    return getattr(quantized_ops, qtype)

class QuantizedLayer():
    """ Quantization functions
    """
    def __init__(self, *args, qfx=None, **kwargs):
        super().__init__(*args, **{**kwargs, "bias": False})
        assert qfx is not None, "Quantization function is not set for Quantized Layer"
        self.qfx = str2layer(qfx)

    def get_quantized_weights(self):
        with torch.no_grad():
            delta_q = self.qfx(self.weight) - self.weight

        quantized_weight = self.weight + delta_q
        return quantized_weight


class QuantizedLinear(QuantizedLayer, nn.Linear):
    def forward(self, nx):
        nx = F.linear(nx, self.get_quantized_weights(), None)
        return nx


class QuantizedConv2d(QuantizedLayer, nn.Conv2d):
    def forward(self, nx):
        nx = F.conv2d(
            nx, self.get_quantized_weights(), None, self.stride,
            self.padding, self.dilation, self.groups
        )
        return nx


class QuantizedActivation(nn.Module):
    def __init__(self, qfx=None):
        super().__init__()
        assert qfx is not None, "Quantization function is not set for QuantizedActivation"
        self.qfx = str2layer(qfx)

    def forward(self, nx):
        nx = self.qfx(nx)
        return nx


""" Pruning Layers using:
 - a Straight-Through-Estimator (STE) (gradients are computed on ALL the weights, pruning mask is NOT fixed)
 - a NonSticky (gradients are computed on the UNPRUNED weights, pruning mask is NOT fixed)
 - a Sticky (gradients are computed on the UNPRUNED weights, pruning mask IS fixed)
 - a Probabilistic approach making probability derivable
 - a Probabilistic approach training probability through an STE
"""
class PrunedLayer():
    def __init__(self, *args, soft, rescale=False, **kwargs):
        super().__init__(*args, **{**kwargs, "bias": False})
        self.th = nn.Parameter(torch.tensor([-1], dtype=self.weight.dtype, device=self.weight.device, requires_grad=False), requires_grad=False)
        self.fixed = nn.Parameter(torch.tensor([False], dtype=torch.bool, device=self.weight.device, requires_grad=False), requires_grad=False)
        self.soft = soft
        self.rescale = rescale
        scale_shape = (self.weight.shape[0],) + (1,)*(len(self.weight.shape)-1)
        self.scale = nn.Parameter(torch.ones(scale_shape, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False), requires_grad=False)
        self.mask = nn.Parameter(torch.ones_like(self.weight, requires_grad=False), requires_grad=False)

    @torch.no_grad()
    def get_scale(self, mask):
        if self.rescale:
            wabs = self.weight.abs()
            axis_mean = tuple(range(1,len(mask.shape)))
            scale = ((wabs*mask).mean(dim=axis_mean, keepdim=True) + 1e-6) / (wabs.mean(dim=axis_mean, keepdim=True) + 1e-6)
            scale = 1.01 / (scale + 0.01)
            return scale
        return self.scale

    @torch.no_grad()
    def set_prune(self, th):
        self.th[:] = th
        pruned_weights = (torch.abs(self.weight) > th)
        self.mask[:] = pruned_weights.to(self.mask.dtype)
        self.scale[:] = self.get_scale(self.mask)

    @torch.no_grad()
    def fix(self):
        self.fixed[:] = True
        self.weight[:] = self.get_quantized_weights()

    def get_quantized_weights(self):
        raise NotImplementedError()

    def get_weights(self):
        if self.fixed:
            return self.weight * self.mask
        return self.get_quantized_weights()


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nx, f):
        with torch.no_grad():
            return f(nx)

    @staticmethod
    def backward(ctx, g):
        return g, None


class STEPrunedLayer(PrunedLayer):
    def get_quantized_weights(self):
        with torch.no_grad():
            weight = self.weight
            if self.th > 0:
                if self.soft:
                    qweight = (weight - torch.sign(weight) * self.th) * self.mask
                else:
                    qweight = weight * self.mask
                if self.rescale:
                    qweight = self.scale * qweight
                delta_q = qweight - weight
            else:
                delta_q = 0
        quantized_weight = weight + delta_q
        return quantized_weight


class NonStickyPrunedLayer(PrunedLayer):
    def get_quantized_weights(self):
        if self.th > 0:
            weight = self.weight
            if self.soft:
                qweight = torch.sign(weight) * ((weight.abs()*self.mask) - self.th).clamp_(min=0)
            else:
                qweight = weight * self.mask
            if self.rescale:
                qweight = self.scale * qweight
            return qweight
        return self.weight


class StickyPrunedLayer(NonStickyPrunedLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_handle = self.weight.register_hook(lambda grad, mask=self.mask: grad*mask)

    @torch.no_grad()
    def set_prune(self, th):
        super().set_prune(th)
        self.weight[:] = self.weight * self.mask


class ProbaPrunedLayer(PrunedLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "bias": False})
        self.probas = nn.Parameter(torch.ones_like(self.weight), requires_grad=True)
        self.temperature = nn.Parameter(torch.tensor([1.0], dtype=self.weight.dtype, device=self.weight.device, requires_grad=False), requires_grad=False)

    @torch.no_grad()
    def set_prune(self, offset):
        self.probas.sub_(offset).clamp_(min=0, max=1)
        self.th[:] = self.weight.abs()[self.probas>0].min()

    @torch.no_grad()
    def set_temperature(self, temperature):
        self.temperature[:] = temperature

    def _get_mask(self):
        if self.fixed:
            return self.mask
        eps = 1e-20
        temp = self.temperature
        uniform0 = torch.rand_like(self.probas)
        uniform1 = torch.rand_like(self.probas)
        noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
        return torch.sigmoid((torch.log(self.probas + eps) - torch.log(1.0 - self.probas + eps) + noise) * temp)

    @torch.no_grad()
    def fix(self):
        self.mask[:] = (torch.rand_like(self.probas) < self.probas).to(self.weight.dtype)
        super().fix()

    def get_quantized_weights(self):
        if self.th < 0:
            return self.weight

        if self.training:
            mask = self._get_mask()
            with torch.no_grad():
                self.mask[:] = mask.clone()
        else:
            mask = self.mask

        self.scale[:] = self.get_scale(mask)
        weight = self.weight
        if self.soft:
            qweight = torch.sign(weight) * ((weight.abs()*mask) - self.th).clamp_(min=0)
        else:
            qweight =  weight * mask
        if self.rescale:
            qweight = qweight * self.scale
        return qweight


""" Linear Pruned Layers
"""
class PrunedLinear(nn.Linear):
    def forward(self, nx):
        nx = F.linear(nx, self.get_weights(), None)
        return nx

class STEPrunedLinear(STEPrunedLayer, PrunedLinear):
    pass
class NonStickyPrunedLinear(NonStickyPrunedLayer, PrunedLinear):
    pass
class StickyPrunedLinear(StickyPrunedLayer, PrunedLinear):
    pass
class ProbaPrunedLinear(ProbaPrunedLayer, PrunedLinear):
    pass


""" Conv2d Pruned Layers
"""
class PrunedConv2d(nn.Conv2d):
    def forward(self, nx):
        nx = F.conv2d(
            nx, self.get_weights(), None, self.stride,
            self.padding, self.dilation, self.groups
        )
        return nx

class STEPrunedConv2d(STEPrunedLayer, PrunedConv2d):
    pass
class NonStickyPrunedConv2d(NonStickyPrunedLayer, PrunedConv2d):
    pass
class StickyPrunedConv2d(StickyPrunedLayer, PrunedConv2d):
    pass
class ProbaPrunedConv2d(ProbaPrunedLayer, PrunedConv2d):
    pass
