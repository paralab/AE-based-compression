import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from datasets import register
from utils import to_pixel_samples, make_coord


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def to_function_samples(func):
    """ Convert the function to coord-value pairs.
        func: Tensor, (1, H, W)
    """
    coord = make_coord(func.shape[-2:])
    values = func.view(1, -1).permute(1, 0)  # (H*W, 1)
    return coord, values


def resize_fn_single_channel(func, size):
    """Resize single channel function using bilinear interpolation"""
    # func: (1, H, W)
    func_resized = F.interpolate(func.unsqueeze(0), size=size, mode='bilinear', align_corners=False)
    return func_resized.squeeze(0)


@register('math-function-downsampled')
class MathFunctionDownsampled(Dataset):
    """Wrapper for mathematical function dataset with downsampling for super-resolution training"""

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        func = self.dataset[idx]  # (1, H, W)
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(func.shape[-2] / s + 1e-9)
            w_lr = math.floor(func.shape[-1] / s + 1e-9)
            func = func[:, :round(h_lr * s), :round(w_lr * s)]
            func_down = resize_fn_single_channel(func, (h_lr, w_lr))
            crop_lr, crop_hr = func_down, func
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, func.shape[-2] - w_hr)
            y0 = random.randint(0, func.shape[-1] - w_hr)
            crop_hr = func[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn_single_channel(crop_hr, (w_lr, w_lr))

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_values = to_function_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_values = hr_values[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_values,
            'scale': torch.tensor(s, dtype=torch.float32)  # Add explicit scale information
        }


def resize_fn_3d(img, size):
    """3D resize function using trilinear interpolation"""
    if isinstance(size, int):
        size = (size, size, size)
    return F.interpolate(img.unsqueeze(0), size=size, mode='trilinear', align_corners=False).squeeze(0)


def to_3d_samples(img):
    """Convert 3D image to coordinate-value pairs"""
    D, H, W = img.shape[-3:]
    coord = make_coord_3d((D, H, W))
    values = img.view(-1, D * H * W).permute(1, 0)
    return coord, values


def make_coord_3d(shape, ranges=None, flatten=True):
    """Make 3D coordinates"""
    D, H, W = shape
    if ranges is None:
        ranges = [(-1, 1), (-1, 1), (-1, 1)]
    
    coord_seqs = []
    for i, n in enumerate(shape):
        r0, r1 = ranges[i]
        coord_seqs.append(torch.linspace(r0, r1, n))
    
    coord_z, coord_y, coord_x = torch.meshgrid(*coord_seqs, indexing='ij')
    coord = torch.stack([coord_x, coord_y, coord_z], dim=-1)  # (D, H, W, 3)
    
    if flatten:
        coord = coord.view(-1, 3)  # (D*H*W, 3)
    
    return coord


@register('math-function-3d-downsampled')
class MathFunction3DDownsampled(Dataset):
    """Wrapper for 3D mathematical function dataset with downsampling for super-resolution training"""

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        func = self.dataset[idx]  # (1, D, H, W)
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            d_lr = math.floor(func.shape[-3] / s + 1e-9)
            h_lr = math.floor(func.shape[-2] / s + 1e-9)
            w_lr = math.floor(func.shape[-1] / s + 1e-9)
            func = func[:, :round(d_lr * s), :round(h_lr * s), :round(w_lr * s)]
            func_down = resize_fn_3d(func, (d_lr, h_lr, w_lr))
            crop_lr, crop_hr = func_down, func
        else:
            d_lr = h_lr = w_lr = self.inp_size
            d_hr = h_hr = w_hr = round(self.inp_size * s)
            
            # Random crop from high-resolution function
            x0 = random.randint(0, max(0, func.shape[-3] - d_hr))
            y0 = random.randint(0, max(0, func.shape[-2] - h_hr))
            z0 = random.randint(0, max(0, func.shape[-1] - w_hr))
            crop_hr = func[:, x0: x0 + d_hr, y0: y0 + h_hr, z0: z0 + w_hr]
            crop_lr = resize_fn_3d(crop_hr, (d_lr, h_lr, w_lr))

        if self.augment:
            # 3D augmentations
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5
            # Rotation around axes
            rot_xy = random.random() < 0.5
            rot_xz = random.random() < 0.5
            rot_yz = random.random() < 0.5

            def augment_3d(x):
                if hflip:
                    x = x.flip(-2)  # flip height
                if vflip:
                    x = x.flip(-1)  # flip width
                if dflip:
                    x = x.flip(-3)  # flip depth
                if rot_xy:
                    x = x.transpose(-2, -1)  # rotate in xy plane
                if rot_xz:
                    x = x.transpose(-3, -1)  # rotate in xz plane
                if rot_yz:
                    x = x.transpose(-3, -2)  # rotate in yz plane
                return x

            crop_lr = augment_3d(crop_lr)
            crop_hr = augment_3d(crop_hr)

        hr_coord, hr_values = to_3d_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_values = hr_values[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-3]  # depth
        cell[:, 1] *= 2 / crop_hr.shape[-2]  # height
        cell[:, 2] *= 2 / crop_hr.shape[-1]  # width

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_values,
            'scale': torch.tensor([s, s, s])  # 3D scale
        }
