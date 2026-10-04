"""Train TinyLaneSeg from scratch on the PSU-Reservoir polygon masks.

    python train.py --data-dir /path/to/psu_reservoir_dataset --epochs 80
"""
import argparse
import json
import os
import time

import torch

from src.dataset import augment, build_cache, load_split, normalize
from src.metrics import confusion, scores_from_confusion, seg_loss
from src.model import TinyLaneSeg, count_params


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, x, y, device, bs=16, num_classes=3):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total, n = 0.0, 0
    for i in range(0, len(x), bs):
        xb = normalize(x[i:i + bs].to(device))
        yb = y[i:i + bs].to(device)
        logits = model(xb)
        total += seg_loss(logits, yb).item() * len(xb)
        n += len(xb)
        cm += confusion(logits.argmax(1), yb, num_classes).cpu()
    return total / n, scores_from_confusion(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache", default="data/cache_256x448.npz")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="weights")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if not os.path.exists(args.cache):
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        print("building mask cache ...")
        build_cache(args.data_dir, args.cache)

    device = get_device()
    xtr, ytr, _ = load_split(args.cache, "train")
    xva, yva, _ = load_split(args.cache, "val")
    print(f"device={device} train={len(xtr)} val={len(xva)}")

    model = TinyLaneSeg().to(device)
    print(f"params: {count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = (len(xtr) + args.bs - 1) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(xtr))
        run, seen = 0.0, 0
        for i in range(0, len(xtr), args.bs):
            idx = perm[i:i + args.bs]
            xb = xtr[idx].to(device).float() / 255.0
            yb = ytr[idx].to(device)
            xb, yb = augment(xb, yb)
            xb = normalize(xb * 255.0)
            loss = seg_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(idx)
            seen += len(idx)

        val_loss, val = evaluate(model, xva, yva, device)
        rec = {"epoch": epoch, "train_loss": run / seen, "val_loss": val_loss,
               "val_miou": val["miou"], "val_iou_lane": val["iou"][1],
               "lr": sched.get_last_lr()[0], "time_s": time.time() - t0}
        history.append(rec)
        flag = ""
        if val["miou"] > best:
            best = val["miou"]
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            flag = " *"
        print(f"ep {epoch:3d} | train {rec['train_loss']:.4f} | val {val_loss:.4f} | "
              f"mIoU {val['miou']:.4f} | lane IoU {val['iou'][1]:.4f} | {rec['time_s']:.1f}s{flag}",
              flush=True)
        with open("results/history.json", "w") as f:
            json.dump(history, f, indent=1)

    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
    print(f"best val mIoU: {best:.4f}")


if __name__ == "__main__":
    main()
