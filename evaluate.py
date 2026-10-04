"""Evaluate a trained model on the held-out test blocks.

    python evaluate.py --weights weights/best.pt
"""
import argparse
import json

import torch

from src.dataset import CLASS_NAMES, load_split, normalize
from src.metrics import confusion, scores_from_confusion
from src.model import TinyLaneSeg
from train import get_device


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache_256x448.npz")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = get_device()
    model = TinyLaneSeg().to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    x, y, names = load_split(args.cache, args.split)

    nc = len(CLASS_NAMES)
    cm = torch.zeros(nc, nc, dtype=torch.long)
    per_image_lane_iou = []
    for i in range(0, len(x), 16):
        pred = model(normalize(x[i:i + 16].to(device))).argmax(1).cpu()
        gt = y[i:i + 16]
        cm += confusion(pred, gt, nc)
        for p, g in zip(pred, gt):
            inter = ((p == 1) & (g == 1)).sum().item()
            union = ((p == 1) | (g == 1)).sum().item()
            per_image_lane_iou.append(inter / union if union else 1.0)

    s = scores_from_confusion(cm)
    s["split"] = args.split
    s["num_images"] = len(x)
    s["class_names"] = CLASS_NAMES
    s["confusion_matrix"] = cm.tolist()
    s["lane_iou_per_image_min"] = min(per_image_lane_iou)
    s["lane_iou_per_image_median"] = sorted(per_image_lane_iou)[len(per_image_lane_iou) // 2]
    s["lane_iou_per_image"] = dict(zip(names, per_image_lane_iou))

    print(f"{args.split}: {len(x)} images")
    print(f"{'class':<12}{'IoU':>8}{'Dice':>8}{'Prec':>8}{'Recall':>8}")
    for k, n in enumerate(CLASS_NAMES):
        print(f"{n:<12}{s['iou'][k]:8.4f}{s['dice'][k]:8.4f}{s['precision'][k]:8.4f}{s['recall'][k]:8.4f}")
    print(f"mIoU {s['miou']:.4f} | mDice {s['mdice']:.4f} | pixel acc {s['pixel_acc']:.4f}")
    print(f"lane IoU per image: median {s['lane_iou_per_image_median']:.4f}, "
          f"min {s['lane_iou_per_image_min']:.4f}")
    with open(f"results/metrics_{args.split}.json", "w") as f:
        json.dump(s, f, indent=1)


if __name__ == "__main__":
    main()
