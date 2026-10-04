"""Run inference and save before/after snapshots.

    # test-set snapshots (input | ground truth | prediction)
    python predict.py --snapshots 6
    # any image
    python predict.py --image some_frame.jpg --out results/my_pred.png
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from src.dataset import load_image, load_split, normalize
from src.model import TinyLaneSeg
from train import get_device

# RGB colours for background / lane / sideway
PALETTE = np.array([[0, 0, 0], [255, 40, 40], [40, 200, 60]], np.uint8)


def overlay(img, mask, alpha=0.5):
    color = PALETTE[mask]
    out = img.copy()
    m = mask > 0
    out[m] = (img[m] * (1 - alpha) + color[m] * alpha).astype(np.uint8)
    return out


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (8 + 11 * len(text), 26), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


@torch.no_grad()
def predict_batch(model, x_uint8, device):
    return model(normalize(x_uint8.to(device))).argmax(1).cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--cache", default="data/cache_256x448.npz")
    ap.add_argument("--image")
    ap.add_argument("--out", default="results/prediction.png")
    ap.add_argument("--snapshots", type=int, default=6)
    args = ap.parse_args()

    device = get_device()
    model = TinyLaneSeg().to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device))

    if args.image:
        img = load_image(args.image)
        x = torch.from_numpy(img).permute(2, 0, 1)[None]
        pred = predict_batch(model, x, device)[0]
        both = np.concatenate([label(img, "input"), label(overlay(img, pred), "prediction")], 1)
        cv2.imwrite(args.out, cv2.cvtColor(both, cv2.COLOR_RGB2BGR))
        print("saved", args.out)
        return

    x, y, names = load_split(args.cache, "test")
    with open("results/metrics_test.json") as f:
        per_img = json.load(f)["lane_iou_per_image"]
    # evenly spread frames across the test blocks, plus the worst case
    picks = list(np.linspace(0, len(x) - 1, args.snapshots - 1).astype(int))
    worst = names.index(min(per_img, key=per_img.get))
    picks.append(worst)

    os.makedirs("results/snapshots", exist_ok=True)
    rows = []
    for i in picks:
        img = x[i].permute(1, 2, 0).numpy()
        pred = predict_batch(model, x[i:i + 1], device)[0]
        gt = y[i].numpy().astype(np.uint8)
        iou = per_img[names[i]]
        row = np.concatenate([
            label(img, f"input  {names[i]}"),
            label(overlay(img, gt), "ground truth"),
            label(overlay(img, pred), f"prediction  lane IoU {iou:.3f}"),
        ], 1)
        cv2.imwrite(f"results/snapshots/{os.path.splitext(names[i])[0]}.png",
                    cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        rows.append(row)
    grid = np.concatenate(rows, 0)
    cv2.imwrite("results/snapshots_grid.jpg", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("saved results/snapshots_grid.jpg and", len(rows), "single snapshots")


if __name__ == "__main__":
    main()
