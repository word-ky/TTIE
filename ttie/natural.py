"""Image-only COCO pilot loading and fixed, metadata-independent views."""

from pathlib import Path

from PIL import Image
import torch
from torch.nn import functional as F


CONDITIONS = ('clean', 'homogeneous_dark', 'homogeneous_bright', 'left_right',
              'quadrants', 'smooth_gradient')
VIEW_NAMES = ('full', 'top_left', 'top_right', 'bottom_left', 'bottom_right')


def resize_shorter(image, side):
    h, w = image.shape[-2:]
    size = (side, int(w * side / h)) if h <= w else (int(h * side / w), side)
    return F.interpolate(image, size=size, mode='bicubic', align_corners=False, antialias=True)


def load_image(path: Path):
    with Image.open(path) as source:
        rgb = source.convert('RGB')
        pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        image = pixels.reshape(rgb.height, rgb.width, 3).permute(2, 0, 1)[None].float() / 255
    return resize_shorter(image, 320).clamp(0, 1)


def views(image):
    h, w = image.shape[-2:]
    return (image, image[..., :h//2, :w//2], image[..., :h//2, w//2:],
            image[..., h//2:, :w//2], image[..., h//2:, w//2:])


def clip_pixels(image, size=224, mean=(.48145466, .4578275, .40821073),
                std=(.26862954, .26130258, .27577711)):
    batches = []
    for view in views(image):
        resized = resize_shorter(view, size)
        h, w = resized.shape[-2:]
        top, left = round((h-size)/2), round((w-size)/2)
        batches.append(resized[..., top:top+size, left:left+size])
    batch = torch.cat(batches)
    return (batch - batch.new_tensor(mean)[None, :, None, None]) / batch.new_tensor(std)[None, :, None, None]


def degrade(clean, condition):
    """Offline synthesis only; condition/gain never enter the scoring API."""
    h, w = clean.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h, device=clean.device),
                          torch.arange(w, device=clean.device), indexing='ij')
    if condition == 'clean':
        gain = torch.ones_like(x, dtype=clean.dtype)
    elif condition == 'homogeneous_dark':
        gain = torch.full_like(x, .45, dtype=clean.dtype)
    elif condition == 'homogeneous_bright':
        gain = torch.full_like(x, 1.55, dtype=clean.dtype)
    elif condition == 'left_right':
        gain = torch.where(x < w//2, .45, 1.55)
    elif condition == 'quadrants':
        gain = torch.where(((x >= w//2).int() + (y >= h//2).int()) % 2 == 0, .45, 1.55)
    elif condition == 'smooth_gradient':
        gain = .45 + 1.1*x.to(clean.dtype)/(w-1)
    else:
        raise ValueError(condition)
    return (clean * gain[None, None]).clamp(0, 1)
