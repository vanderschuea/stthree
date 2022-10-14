import kornia.augmentation as k_augmentation
from kornia.augmentation.container import AugmentationSequential
import torch.nn as nn
import torch.nn.functional as F
import torch

from ..parsers import get_class

class Normalize(nn.Module):
    def __init__(self, mean, std, scaling: float) -> None:
        super().__init__()
        # Make them parameter to move them automatically to gpu if required!
        self.mean = nn.Parameter(torch.tensor(mean).reshape(1,-1,1,1)*scaling, requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).reshape(1,-1,1,1)*scaling, requires_grad=False)

    def forward(self, nx):
        return (nx-self.mean)/self.std


class ToFloat(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, nx):
        return nx.to(float)


class WrapTransform(nn.Module):
    def __init__(self, transform, apply_to) -> None:
        super().__init__()
        self.transform = transform
        if type(apply_to) == str:
            apply_to = list(apply_to)
        self.apply_to = apply_to

    def forward(self, nx):
        nx_copy = nx.copy()
        for key in self.apply_to:
            if key not in nx:
                print(f"Warning missing {key} key")
            else:
                nx_copy[key] = self.transform(nx[key])
        return nx_copy


class KorniaWrapper(nn.Module):  # FIXME: make this a normal class or not?
    def __init__(self, augmentations):
        super().__init__()
        self.augmentations = [
            get_class(augmentation_params, k_augmentation) for augmentation_params in augmentations
        ]
        self.sequence = None
        self.keys = None
        self.supported_types = {"image": "input", "mask": "mask", "bboxes": "bbox_xywh", "keypoints": "keypoints"}
        # TODO: add support for polygons

    def _check_type(self, key):
        for stype in self.supported_types.keys():
            if stype in key:
                return True
        return False

    def _get_kornia_type(self, key):
        for stype in self.supported_types.keys():
            if stype in key:
                return self.supported_types[stype]
        return None

    def _flatten_shape(self, key, batch_points):
        if "mask" in key:
            # Kornia changes mask shape to account for possible multiple instances
            # but doesn't change it back
            N,H,W = batch_points.shape
            return batch_points.reshape(N,1,H,W)

        if "bboxes" not in key and "keypoints" not in key:
            return batch_points

        mx_len = 0
        for points in batch_points:
            mx_len = max(len(points), mx_len)

        tensor_points = []
        for points in batch_points:
            if "keypoints" in key:
                new_points = []
                for coords in points:
                    coords = [coord for i, coord in enumerate(coords) if (i+1)%3 != 0]
                    new_points.append(coords)
                points = new_points

            points = torch.tensor(points)
            points = F.pad(points, (0,0,0,mx_len-len(points)))
            if "keypoints" in key:
                points = points.reshape(-1, 2)

            tensor_points.append(points.unsqueeze_(0))

        tensor_points = torch.cat(tensor_points)
        return tensor_points

    def _unflatten_shape(self, key, tensor_points, batch_points):
        if "mask" in key:
            # Kornia changes mask shape to account for possible multiple instances
            # but doesn't change it back
            return tensor_points.reshape(*batch_points.shape)

        if "bboxes" not in key and "keypoints" not in key:
            return tensor_points

        tbatch_points = []
        for tpoints, points in zip(tensor_points, batch_points):
            if "keypoints" in key:
                # Switch from (#coords_per_instance*(#instance+padding), 2) to
                # (#instance, 2*#coords_per_instance)
                n_coords_per_instance = len(points[0])//3
                tpoints = tpoints.reshape(-1, n_coords_per_instance*2)
            tpoints = tpoints[:len(points)]

            if "keypoints" in key:
                new_tpoints = []
                for tcoords, coords in zip(tpoints, points):
                    tcoords = [
                        tcoords[i-i//3].item() if (i+1)%3!=0 else coords[i] for i in range(len(coords))
                    ]
                    new_tpoints.append(tcoords)
                tpoints = new_tpoints
            tbatch_points.append(tpoints)
        return tbatch_points

    def forward(self, nx):
        if len(self.augmentations)==0:
            return nx

        if self.sequence is None:
            self.keys = [key for key in nx.keys() if self._check_type(key)]
            kornia_keys = [self._get_kornia_type(key) for key in self.keys]
            self.sequence = AugmentationSequential(*self.augmentations, data_keys=kornia_keys)


        # FIXME: remove ugly float hack and add module for conversion
        with torch.no_grad():
            inputs = [self._flatten_shape(key, nx[key]) for key in self.keys]
            inputs = [
                (
                    tensor if tensor.dtype in {torch.float16, torch.float32, torch.float64}
                    else tensor.to(torch.float32)
                )
                for tensor in inputs
            ]
            outputs = self.sequence(*inputs)
            if len(inputs) == 1:  # Fix kornia not being consistent
                outputs = [outputs]
            for key, output in zip(self.keys, outputs):
                nx[key] = self._unflatten_shape(key, output, nx[key])
        return nx
