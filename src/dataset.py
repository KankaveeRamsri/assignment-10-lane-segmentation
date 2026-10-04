"""PSU-Reservoir polygon dataset -> semantic masks.

Only the true polygon classes of the dataset are used:
    YOLO class 0 `Lane`    -> mask label 1
    YOLO class 4 `Sideway` -> mask label 2
    everything else        -> mask label 0 (background / unlabeled)
Classes 1-3 (`Track_Line_L/C/R`) were originally polylines that were expanded into
thin polygons, so they are ignored as required by the assignment.

Split: the 1000 frames come from one video, so neighbouring frames look almost the
same. A random split would leak near-duplicates into the test set. Instead frames
are grouped into contiguous blocks of 50 and whole blocks go to train/val/test.
"""
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CLASS_NAMES = ["background", "lane", "sideway"]
YOLO_TO_MASK = {4: 2, 0: 1}  # draw order: Sideway first, Lane on top
IMG_H, IMG_W = 256, 448
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

BLOCK = 50
TEST_BLOCKS = {2, 7, 12, 17}  # 200 frames
VAL_BLOCKS = {4, 14}          # 100 frames


def frame_index(path):
    return int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])


def split_of(idx):
    block = (idx - 1) // BLOCK
    if block in TEST_BLOCKS:
        return "test"
    if block in VAL_BLOCKS:
        return "val"
    return "train"


def read_polygons(label_path):
    polys = []
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 7:
                continue
            cls = int(p[0])
            xy = np.array(p[1:], dtype=np.float32).reshape(-1, 2)
            polys.append((cls, xy))
    return polys


def rasterize(polys, h, w):
    mask = np.zeros((h, w), np.uint8)
    for yolo_cls in YOLO_TO_MASK:
        for cls, xy in polys:
            if cls == yolo_cls:
                pts = np.round(xy * [w, h]).astype(np.int32)
                cv2.fillPoly(mask, [pts], YOLO_TO_MASK[yolo_cls])
    return mask


def load_image(path, h=IMG_H, w=IMG_W):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def build_cache(data_dir, cache_path):
    """Resize all images and rasterize masks once; store as a single .npz."""
    imgs = sorted(glob.glob(os.path.join(data_dir, "images", "*", "*.jpg")))
    X, Y, names, splits = [], [], [], []
    for p in imgs:
        sub = os.path.basename(os.path.dirname(p))
        lab = os.path.join(data_dir, "labels", sub, os.path.splitext(os.path.basename(p))[0] + ".txt")
        X.append(load_image(p))
        Y.append(rasterize(read_polygons(lab), IMG_H, IMG_W))
        names.append(os.path.basename(p))
        splits.append(split_of(frame_index(p)))
    order = np.argsort([frame_index(n) for n in names])
    np.savez_compressed(
        cache_path,
        images=np.stack(X)[order], masks=np.stack(Y)[order],
        names=np.array(names)[order], splits=np.array(splits)[order],
    )


def load_split(cache_path, split):
    d = np.load(cache_path)
    sel = d["splits"] == split
    x = torch.from_numpy(d["images"][sel]).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,H,W
    y = torch.from_numpy(d["masks"][sel]).long()                              # N,H,W
    return x, y, list(d["names"][sel])


def normalize(x_uint8):
    x = x_uint8.float() / 255.0
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x, y):
    """GPU-side augmentation on a batch. x: float [0,1] N,3,H,W ; y: long N,H,W."""
    n = x.shape[0]
    dev = x.device
    # photometric: brightness / contrast / saturation jitter (outdoor sun & shadow)
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.5
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.5
    s = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.6
    gray = x.mean(1, keepdim=True)
    x = (x - gray) * s + gray
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * c + m) * b
    x = x.clamp(0, 1)

    # geometric: horizontal flip + random scale / shift (same transform for mask)
    flip = torch.rand(n, device=dev) < 0.5
    scale = 1 / (1 + torch.rand(n, device=dev) * 0.25)  # zoom-in 1.0 - 1.25x
    tx = (torch.rand(n, device=dev) - 0.5) * 0.2
    ty = (torch.rand(n, device=dev) - 0.5) * 0.2
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = scale * torch.where(flip, -1.0, 1.0)
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="reflection",
                      align_corners=False)[:, 0].long()
    return x, y
