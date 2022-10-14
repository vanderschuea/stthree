import torch
import torch.nn as nn
import math
from ...parsers import get_class
from . import quantized

class ModelConfig():
    ''' Contains the layers used for the model
        if @all is set, its value will override @conv & @act
    '''
    def __init__(self, bn=None, act=None, conv=None, fc=None, pixel_shuffle=None):
        self.conv = lambda *args, **kwargs: get_class(conv, [quantized, torch.nn], *args, **kwargs)

        # TODO: BN not supported yet -> should add instance BN in the future
        self.bn = nn.BatchNorm2d

        self.act = lambda *args, **kwargs: get_class(act, [quantized, torch.nn], *args, **kwargs)

        self.fc = lambda *args, **kwargs: get_class(fc, [quantized, torch.nn], *args, **kwargs)

        self.pixel_shuffle = lambda *args, **kwargs: get_class(pixel_shuffle, [quantized, torch.nn], *args, **kwargs)

@torch.no_grad()
def init_weights(model, cfg):
    params = cfg["network"].get("weight_init", {})

    # This part fixes pytorch buggy default implementation
    act = cfg["network"]["layers"]["act"]["type"]
    act = cfg["network"]["layers"]["act"].get("qfx", act).lower()
    if "leaky" in act:
        neg_slope = 0.01
        nonlin = "leaky_relu"
        sampling = "kaiming"
    elif "relu" in act:
        neg_slope = 0
        nonlin = "relu"
        sampling = "kaiming"
    elif "tanh" in act:
        neg_slope = 0
        nonlin = "tanh"
        sampling = "kaiming"
    else:
        print(f"Activation of type {act} is not supported yet")
    # Divide by sqrt(2) to support pytorch's stupid way of implementing xavier
    gain = nn.init.calculate_gain(nonlin, neg_slope)

    # Override default params
    gamma = params.get("gamma", 1.0)
    momentum  = params.get("momentum", 0.9)
    sampling = params.get("sampling", sampling)
    distribution = params.get("distribution", "normal")
    fan_mode = params.get("fan_mode", "fan_in")
    probas_scale = params.get("probas", 1)
    probas_distribution = params.get("probas_distribution", "constant")
    gain = params.get("gain", gain)

    assert sampling in ["kaiming", "xavier"]
    assert distribution in ["normal", "uniform"]
    assert fan_mode in ["fan_in", "fan_out", "fan_avg"]

    def custom_weights_init(m):
        # This custom part does things by the book and mirrors Keras'
        # implementation instead of the wonky pytorch one
        # Support for depthwise convolutions has also been added
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if isinstance(m, nn.Conv2d):
                ksize = m.kernel_size[0] * m.kernel_size[1]
                ksize = ksize / m.groups
                fan_out = m.out_channels * ksize
                fan_in = m.in_channels * ksize
            else:
                fan_out = m.out_features
                fan_in = m.in_features
            fan_avg = (fan_in + fan_out)/2

            if sampling == "xavier":
                std = gain/math.sqrt(fan_in+fan_out)
            elif sampling == "kaiming":
                fan = {
                    "fan_in": fan_in, "fan_out": fan_out, "fan_avg": fan_avg
                }[fan_mode]
                fan = fan  # FIXME 0.95??
                std = gain/math.sqrt(fan)


            if distribution == "normal":
                m.weight.normal_(0, std)
            else:
                limit = math.sqrt(3)*std
                m.weight.uniform_(-limit, limit)

            if hasattr(m, "probas"):
                if probas_distribution=="constant":
                    m.probas.fill_(probas_scale)
                else:
                    nn.init.kaiming_uniform_(m.probas, a=probas_scale)

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.fill_(gamma)
            m.momentum = momentum

        if hasattr(m, "bias") and hasattr(m.bias, "data"):
            m.bias.zero_()

    if "seed" in params:
        torch.manual_seed(params["seed"])
    model.apply(custom_weights_init)

    # Override weights with pretrained ones if necessary
    pretrained_path = params.get("pretrained", "")
    if pretrained_path != "":
        load_pretrained_weights(model, pretrained_path)


@torch.no_grad()
def load_pretrained_weights(model, pretrained_path):
    pretrained_params = torch.load(pretrained_path, map_location="cpu")
    if pretrained_path.endswith(".ckpt"):
        all_params = pretrained_params["state_dict"]
        pretrained_params = {}
        for wname, w in  all_params.items():
            if wname.endswith(".total_ops") or wname.endswith(".total_params"):
                continue
            if wname.startswith("model."):
                pretrained_params[wname[6:]] = w

    print(f"Using pretrained weights from {pretrained_path}")
    if not extensive_pretrained_match(model, pretrained_params):
        exit(-1)

# Adds compatibility for loading old ResNet-50 pretrained weights
def extensive_pretrained_match(model, pretrained_params):
    for n_try in range(2):
        try:
            load_result = model.load_state_dict(pretrained_params)#, strict=True)
            if len(load_result.missing_keys) == 0 and len(load_result.unexpected_keys) == 0:
                break
            print("WARNING: missing or unexpected keys found")
            print("Missing:\n", load_result.missing_keys)
            print("Unexpected:\n", load_result.unexpected_keys)
        except RuntimeError as e:
            print(e)
            if n_try > 0:
                return False
        original_params = model.state_dict()
        print(len(pretrained_params), len(original_params))
        if len(pretrained_params) != len(original_params):
            print("Not the same number of parameters, will try to match as close as possible")

        if n_try > 0:
            return True
        print("WARNING: will try to match keys in a different fashion")

        # Batchnorm compatibility layer for pytorch < 0.4.1
        original_keys = list(original_params.keys())
        pretrained_keys = list(pretrained_params.keys())
        if len([key for key in pretrained_keys if ".num_batches_tracked" in key]) == 0:
            original_keys = [key for key in original_keys if ".num_batches_tracked" not in key]
            # batchnorm.weight        -> bn1.running_mean
            # batchnorm.bias          -> bn1.running_var
            # batchnorm.running_mean  -> bn1.weight
            # batchnorm.running_var   -> bn1.bias
            def _swap_bn_keys(key, orig_key):
                # downsample.1 is there for very specific pytorch bullshittery
                if ".bn" not in orig_key and ".batchnorm" not in orig_key:
                    return key
                if key.rsplit('.')[-1] == orig_key.rsplit('.')[-1]:
                    return key
                if key.endswith(".running_mean"):
                    return key.replace(".running_mean", ".weight")
                elif key.endswith(".running_var"):
                    return key.replace(".running_var", ".bias")
                elif key.endswith(".weight"):
                    return key.replace(".weight", ".running_mean")
                elif key.endswith(".bias"):
                    return key.replace(".bias", ".running_var")
            pretrained_keys = [_swap_bn_keys(key, orig_key) for key, orig_key in zip(pretrained_keys, original_keys)]


        for original_key, pretrained_key in zip(original_keys, pretrained_keys):
            original_val = original_params[original_key]
            pretrained_val = pretrained_params[pretrained_key]
            if original_val.shape != pretrained_val.shape or original_key.rsplit('.')[-1] != pretrained_key.rsplit('.')[-1]:
                print(f"Failed on key '{pretrained_key}'/'{original_key}': {pretrained_val.shape} instead of {original_val.shape}")
            else:
                original_params[original_key] = pretrained_val
        pretrained_params = original_params
    return True