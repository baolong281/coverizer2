# %%

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from networks.mat import Generator
import base64
import glob
import os
import random
import re
from http import HTTPStatus
from io import BytesIO
from typing import Dict, List, NamedTuple, Optional, Tuple

import click
import cv2
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel

import dnnlib
import legacy


pyspng = None


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name,
                   tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(
                tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


class Inpainter:
    def __init__(self,
                 network_pkl,
                 resolution=512,
                 truncation_psi=1,
                 noise_mode='const',
                 sdevice='cpu'
                 ):
        self.resolution = resolution
        self.truncation_psi = truncation_psi
        self.noise_mode = noise_mode
        print(f'Loading networks from: {network_pkl}')
        self.device = torch.device(sdevice)
        with dnnlib.util.open_url(network_pkl) as f:
            G_saved = (
                legacy.load_network_pkl(f)
                ['G_ema']
                .to(self.device)
                .eval()
                .requires_grad_(False))  # type: ignore
        net_res = 512 if resolution > 512 else resolution
        self.G = (
            Generator(
                z_dim=512,
                c_dim=0,
                w_dim=512,
                img_resolution=net_res,
                img_channels=3
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        copy_params_and_buffers(G_saved,  self.G, require_all=True)

    def generate_images2(
        self,
        dpath: List[PIL.Image.Image],
        mpath: List[Optional[PIL.Image.Image]],
        seed: int = 42,
    ):
        """
        Generate images using pretrained network pickle.
        """
        resolution = self.resolution
        truncation_psi = self.truncation_psi
        noise_mode = self.noise_mode
        # seed = 240  # pick up a random number

        def seed_all(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        if seed is not None:
            seed_all(seed)

        # no Labels.
        label = torch.zeros([1,  self.G.c_dim], device=self.device)

        def read_image(image):
            image = np.array(image)
            if image.ndim == 2:
                image = image[:, :, np.newaxis]  # HW => HWC
                image = np.repeat(image, 3, axis=2)
            image = image.transpose(2, 0, 1)  # HWC => CHW
            image = image[:3]
            return image
        if resolution != 512:
            noise_mode = 'random'
        results = []
        with torch.no_grad():
            for i, (ipath, m) in enumerate(zip(dpath, mpath)):
                if seed is None:
                    seed_all(i)

                image = read_image(ipath)
                image = (torch.from_numpy(image).float().to(
                    self. device) / 127.5 - 1).unsqueeze(0)

                mask = np.array(m).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask).float().to(
                    self. device).unsqueeze(0).unsqueeze(0)

                z = torch.from_numpy(np.random.randn(
                    1,  self.G.z_dim)).to(self.device)
                output = self.G(image, mask, z, label,
                                truncation_psi=truncation_psi, noise_mode=noise_mode)
                output = (output.permute(0, 2, 3, 1) * 127.5 +
                          127.5).round().clamp(0, 255).to(torch.uint8)
                output = output[0].cpu().numpy()
                results.append(PIL.Image.fromarray(output, 'RGB'))

        return results


# if __name__ == "__main__":
#     generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
def mask_to_alpha(img, mask):
    img = img.copy()
    img.putalpha(mask)
    return img


def blend(src, target, mask):
    mask = np.expand_dims(mask, axis=-1)
    result = (1-mask) * src + mask * target
    return Image.fromarray(result.astype(np.uint8))


def pad(img, size=(128, 128), tosize=(512, 512), border=1):
    if isinstance(size, float):
        size = (int(img.size[0] * size), int(img.size[1] * size))
    # remove border
    w, h = tosize

    new_img = Image.new('RGBA', (w, h))

    rimg = img.resize(size, resample=Image.Resampling.NEAREST)
    rimg = ImageOps.crop(rimg, border=border)
    tw, th = size
    tw, th = tw - border*2, th - border*2
    tc = ((w-tw)//2, (h-th)//2)

    new_img.paste(rimg, tc)
    mask = Image.new('L', (w, h))
    white = Image.new('L', (tw, th), 255)
    mask.paste(white, tc)

    if 'A' in rimg.getbands():
        mask.paste(rimg.getchannel('A'), tc)
    return new_img, mask


def b64_to_img(b64):
    return Image.open(BytesIO(base64.b64decode(b64)))


def img_to_b64(img):
    with BytesIO() as f:
        img.save(f, format='PNG')
        return base64.b64encode(f.getvalue()).decode('utf-8')


class Predictor:
    def __init__(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.models = {
            # actually laion300k+laion1200k(opmasked)
            "places2": Inpainter(
                network_pkl='models/Places_512_FullData+LAION300k+OPM1200k.pkl',
                resolution=512,
                truncation_psi=1.,
                noise_mode='const',
            ),

        }

    # The arguments and types the model takes as input

    def predict(
        self,
        img: Image.Image,
        tosize=(512, 512),
        border=5,
        seed=42,
        size=0.5,
        model='places2',
    ) -> Image:
        i, m = pad(
            img,
            size=size,  # (328, 328),
            tosize=tosize,
            border=border
        )
        """Run a single prediction on the model"""
        imgs = self.models[model].generate_images2(
            dpath=[i.resize((512, 512), resample=Image.Resampling.NEAREST)],
            mpath=[m.resize((512, 512), resample=Image.Resampling.NEAREST)],
            seed=seed,
        )
        img_op_raw = imgs[0].convert('RGBA')
        img_op_raw = img_op_raw.resize(
            tosize, resample=Image.Resampling.NEAREST)
        inpainted = img_op_raw.copy()

        # paste original image to remove inpainting/scaling artifacts
        inpainted = blend(
            i,
            inpainted,
            1-(np.array(m) / 255)
        )
        minpainted = mask_to_alpha(inpainted, m)
        return inpainted, minpainted,  ImageOps.invert(m)

    def predict_tiled(
        self,
        img: Image.Image,
        tosize=(512, 512),
        border=5,
        seed=42,
        size=0.5,
        model='places2',
    ) -> Image:

        i, morig = pad(
            img,
            size=size,  # (328, 328),
            tosize=tosize,
            border=border
        )
        i.putalpha(morig)
        img = i
        # img.save('0.png')
        assert img.width == img.height
        assert img.width > 512 and img.width <= 512*2

        def tile_coords(image, n=2, tile_size=512):
            assert image.width == image.height
            offsets = np.linspace(0, image.width - tile_size, n).astype(int)
            for i in range(n):
                for j in range(n):
                    left = offsets[j]
                    upper = offsets[i]
                    right = left + tile_size
                    lower = upper + tile_size
                    # tile = image.crop((left, upper, right, lower))
                    yield [left, upper, right, lower]

        for ix, tc in enumerate(tile_coords(img, n=2)):
            i = img.crop(tc)
            # i.save(f't{ix}.png')
            m = i.getchannel('A')

            """Run a single prediction on the model"""
            imgs = self.models[model].generate_images2(
                dpath=[i.resize((512, 512), resample=Image.Resampling.NEAREST)],
                mpath=[m.resize((512, 512), resample=Image.Resampling.NEAREST)],
                seed=seed,
            )
            img_op_raw = imgs[0].convert('RGBA')
            # img_op_raw = img_op_raw.resize(tosize, resample=Image.Resampling.NEAREST)
            inpainted = img_op_raw.copy()

            # paste original image to remove inpainting/scaling artifacts
            inpainted = blend(
                i,
                inpainted,
                1-(np.array(m) / 255)
            )
            # inpainted.save(f't{ix}_op.png')
            minpainted = mask_to_alpha(inpainted, m)
            # continue with partially inpainted image
            # since the tiles overlap, the next tile will contain (possibly inpainted) parts of the previous tile
            img.paste(inpainted, tc)

        # restore original alpha channel
        img.putalpha(morig)
        return img.convert('RGB'), img,  ImageOps.invert(img.getchannel('A'))
predictor = Predictor()

# %%


def _outpaint(img, tosize, border, seed, size, model, tiled):
    if tiled:
        img_op = predictor.predict_tiled(
            img,
            border=border,
            seed=seed,
            tosize=(tosize, tosize),
            size=float(size),
            model=model,
        )
    else:
        img_op = predictor.predict(
            img,
            border=border,
            seed=seed,
            tosize=(tosize, tosize),
            size=float(size),
            model=model,
        )
    return img_op
# %%

