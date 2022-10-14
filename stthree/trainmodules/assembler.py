import copy
import torch.nn as nn
import torchmetrics

from ..parsers import get_class
from ..preprocessors.wrappers import WrapTransform, KorniaWrapper

from ..models import backbones
from ..models import structures
from ..models import losses as model_losses
from ..models.layers import ModelConfig
from .. import preprocessors
import kornia.augmentation as k_augmentation



def get_model(cfg):
    # Get different layer types (TODO: rename layer_classes -> layers at some point)
    layer_classes = ModelConfig(**cfg["network"]["layers"])

    # Make backbone
    backbone = get_class(cfg["network"]["backbone"], backbones, layer_classes=layer_classes)

    # Integrate backbone into complete model
    structure = get_class(
        cfg["network"]["structure"], structures, layer_classes=layer_classes,
        backbone=backbone,
    )

    return structure

def get_preprocessors(cfg):
    transforms = []
    for transform_params in cfg.get("preprocessors", []):
        apply_to = transform_params["apply_to"]
        transform_params = copy.deepcopy(transform_params)  # Never change original config
        del transform_params["apply_to"]
        transform = get_class(transform_params, [preprocessors, k_augmentation])
        transforms.append(WrapTransform(transform=transform, apply_to=apply_to))

    return nn.Sequential(*transforms), KorniaWrapper(cfg.get("augmentations", []))


def _get_metric_from(params_list, module, list_to_tensor):
    metrics = []
    for metric_params in params_list:
        metric_params = copy.deepcopy(metric_params)
        apply_to = metric_params["apply_to"]
        del metric_params["apply_to"]
        if "name" in metric_params:
            name = metric_params["name"]
            del metric_params["name"]
        else:
            name = metric_params["type"].lower()
        metric = get_class(metric_params, module, list_to_tensor=list_to_tensor)
        metric._apply_to = apply_to
        metric._name = name
        metrics.append(metric)
    return metrics


def get_losses(cfg):
    losses = _get_metric_from(cfg["network"].get("losses", []), [model_losses, nn], list_to_tensor=True)

    metrics = _get_metric_from(cfg["network"].get("metrics", []), torchmetrics, list_to_tensor=False)

    omlosses = []
    for omloss_params in cfg["network"].get("omlosses", []):
        omlosses.append(lambda **kwargs: get_class(omloss_params, model_losses, **kwargs))

    return losses, metrics, omlosses
