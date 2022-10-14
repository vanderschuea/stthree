import cv2
import torch
import numpy as np
from abc import ABC



class CVTransform(ABC):
    """ Abstract class for CVTransforms

    Forces every CVTransform to explicitely implement every method in order
    to limit the amount of possible bugs

    Attributes:
        PASS_THROUGH: dummy function, usefull to implement required method in
            one line of code.
    """
    @staticmethod
    def PASS_THROUGH(nx, arr):
        return arr

    def image(self, nx, arr):
        return NotImplemented

    def mask(self, nx, arr):
        return NotImplemented

    def keypoints(self, nx, arr):
        return NotImplemented

    def bboxes(self, nx, arr):
        return NotImplemented

    def polygons(self, nx, arr):
        return NotImplemented

    def classification(self, nx, arr):
        return NotImplemented

    def shape(self, nx, arr):
        return NotImplemented

    def __call__(self, nx):
        new_dict = {}
        for key, arr in nx.items():
            if "image" in key:
                new_dict[key] = self.image(nx, arr)
            elif "mask" in key:
                new_dict[key] = self.mask(nx, arr)
            elif "keypoints" in key:
                new_dict[key] = self.keypoints(nx, arr)
            elif "bboxes" in key:
                new_dict[key] = self.bboxes(nx, arr)
            elif "polygons" in key:
                new_dict[key] = self.polygons(nx, arr)
            elif "class" in key:
                new_dict[key] = self.classification(nx, arr)
            elif "shape" in key:
                new_dict[key] = self.shape(nx, arr)
            else:
                raise ValueError(f"CVTransform does not yet support {key} formatted data")
        return new_dict

def pass_through(*args):
    keys = {*args}
    """ Decorator to automatically pass through some of the keys

    Usage:
        Extend CVTransform, decorate it with passthrough and the list of keys
        you want to pass through without modification. This allows forces explicit
        function naming

    Example:
        @pass_through('image', 'mask', 'keypoints', 'bboxes', 'polygons', 'classification', 'shape')
        class DummyCVTransform(CVTransform):
            pass
    """
    def cvtransform_wrapper(CVTransform_Class):
        class CVTransformWrapper(CVTransform_Class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __getattribute__(self, s):
                if s in keys:
                    return CVTransform.PASS_THROUGH
                return super().__getattribute__(s)

        return CVTransformWrapper  # decoration ends here
    return cvtransform_wrapper


@pass_through("classification")
class Rescale(CVTransform):
    def __init__(self, vdim: int = None, hdim: int = None, scale: float = None):
        assert ((vdim is not None) + (hdim is not None) + (scale is not None)) == 1, 'One and only one of hdim, vdim or scale needs to be set'
        if vdim is not None:
            self.get_scale = lambda arr_shape: vdim/arr_shape[0]
            self.get_dims = lambda arr_shape: (round(arr_shape[1]*self.get_scale(arr_shape)), vdim)
        elif hdim is not None:
            self.get_scale = lambda arr_shape: hdim/arr_shape[1]
            self.get_dims = lambda arr_shape: (hdim, round(arr_shape[0]*self.get_scale(arr_shape)))
        elif scale is not None:
            self.get_scale = lambda arr_shape: scale
            self.get_dims = lambda arr_shape: (round(arr_shape[1]*scale), round(arr_shape[0]*scale))

        self.classification = CVTransform.PASS_THROUGH

    def image(self, nx, arr):
        return cv2.resize(arr, self.get_dims(nx["shape"]), interpolation=cv2.INTER_LINEAR)

    def mask(self, nx, arr):
        return cv2.resize(arr, self.get_dims(nx["shape"]), interpolation=cv2.INTER_NEAREST)

    def keypoints(self, nx, arr):
        new_arr = []
        scale = self.get_scale(nx["shape"])
        for keypoint_arr in arr:
            keypoint_arr = [  # FIXME: should we round keypoints??
                coord*scale if ic%3 != 2 else coord for ic, coord in enumerate(keypoint_arr)
            ]

            new_arr.append(keypoint_arr)
        return new_arr
    def bboxes(self, nx, arr):
        new_arr = []
        scale = self.get_scale(nx["shape"])
        for bbox_arr in arr:
            new_arr.append([coord*scale for coord in bbox_arr])
        return new_arr

    def polygons(self, nx, arr):
        new_arr = []
        scale = self.get_scale(nx["shape"])
        for polygons_arr in arr:
            new_arr.append([
                [coord*scale for coord in poly_arr]
                for poly_arr in polygons_arr
            ])
        return new_arr

    def shape(self, nx, arr):
        shape = self.get_dims(nx["shape"])
        return (shape[1], shape[0])


@pass_through("classification")
class RandomCropAndPadding(CVTransform):
    def __init__(self, vdim: int, hdim: int):
        self.vdim = vdim
        self.hdim = hdim
        self._reset_jitter()

    def _get_jitter(self, vmargin, hmargin):
        if self.jitter is None:
            vjitter = torch.randint(low=-vmargin//2, high=vmargin//2+1, size=(1,)).item()
            hjitter = torch.randint(low=-hmargin//2, high=hmargin//2+1, size=(1,)).item()
            self.jitter = (vjitter, hjitter)
        return self.jitter

    def _reset_jitter(self):
        self.jitter = None

    def _pad_crop_img(self, arr, shape):
        vmargin, hmargin = shape[0] - self.vdim, shape[1] - self.hdim
        if shape[0] < self.vdim or shape[1] < self.hdim:
            top    = -vmargin//2                if shape[0] < self.vdim else 0
            bottom = -vmargin//2 + (-vmargin)%2 if shape[0] < self.vdim else 0
            left   = -hmargin//2                if shape[1] < self.hdim else 0
            right  = -hmargin//2 + (-hmargin)%2 if shape[1] < self.hdim else 0
            arr = cv2.copyMakeBorder(arr, top, bottom, left, right, cv2.BORDER_CONSTANT)

            shape = arr.shape[:2]
            vmargin, hmargin = shape[0] - self.vdim, shape[1] - self.hdim

        vjitter, hjitter = self._get_jitter(vmargin, hmargin)

        top    = vmargin//2 - vjitter
        bottom = top + self.vdim
        left   = hmargin//2 - hjitter
        right  = left + self.hdim

        return arr[top:bottom, left:right]

    def _get_xy_offsets(self, shape):
        voffset, hoffset = 0, 0
        vmargin, hmargin = shape[0] - self.vdim, shape[1] - self.hdim
        if shape[0] < self.vdim:
            voffset += -vmargin//2
            vmargin = 0
        if shape[1] < self.hdim:
            hoffset += -hmargin//2
            hmargin = 0

        voffset -= vmargin//2
        hoffset -= hmargin//2

        vjitter, hjitter = self._get_jitter(vmargin, hmargin)
        voffset += vjitter
        hoffset += hjitter
        return voffset, hoffset

    def image(self, nx, arr):
        return self._pad_crop_img(arr, nx['shape'])

    def mask(self, nx, arr):
        # TODO: mask padding should be ignore_index padded
        return self._pad_crop_img(arr, nx['shape'])

    def keypoints(self, nx, arr):
        voffset, hoffset = self._get_xy_offsets(nx["shape"])
        offset_keypoint = [
            lambda coord: coord+hoffset,
            lambda coord: coord+voffset,
            lambda coord: coord
        ]
        new_arr = []
        for keypoint_arr in arr:
            keypoint_arr = [
                0 if keypoint_arr[3*ic+2]==0 else offset_keypoint[jc](keypoint_arr[3*ic+jc])
                for ic in range(len(keypoint_arr)//3) for jc in range(3)
            ]
            new_arr.append(keypoint_arr)
        return new_arr

    def bboxes(self, nx, arr):
        voffset, hoffset = self._get_xy_offsets(nx["shape"])
        new_arr = []
        for bbox_arr in arr:
            x, y, w, h = bbox_arr
            new_arr.append([x+hoffset, y+voffset, w, h])
        return new_arr

    def polygons(self, nx, arr):
        voffset, hoffset = self._get_xy_offsets(nx["shape"])
        offset_polygon = [
            lambda coord: coord+hoffset,
            lambda coord: coord+voffset,
        ]
        new_arr = []
        for polygons_arr in arr:
            new_arr.append([
                [offset_polygon[ic%2](coord) for ic, coord in enumerate(poly_arr)]
                for poly_arr in polygons_arr
            ])
        return new_arr


    def shape(self, nx, arr):
        return (self.vdim, self.hdim)

    def __call__(self, nx):
        new_dict = super().__call__(nx)
        self._reset_jitter()
        return new_dict


@pass_through("classification")  # -> FIXME: is this necessary?
class CenterCropAndPadding(RandomCropAndPadding):
    def _get_jitter(self, vmargin, hmargin):
        self.jitter = (0, 0)
        return self.jitter


@pass_through("keypoints", "bboxes", "polygons", "shape")
class ToTensor(CVTransform):
    def image(self, nx, arr):
        # print(arr.dtype)
        # import matplotlib.pyplot as plt
        # plt.imshow(arr)
        # plt.show()
        return torch.from_numpy(arr).permute(2,0,1).unsqueeze_(0)

    def mask(self, nx, arr):
        return torch.from_numpy(arr).unsqueeze_(0)

    def classification(self, nx, arr):
        return torch.tensor(arr).unsqueeze_(0)
