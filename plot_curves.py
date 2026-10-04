"""Plot training curves from results/history.json -> results/loss_curve.png"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results/history.json") as f:
    h = json.load(f)
ep = [r["epoch"] for r in h]
best = max(h, key=lambda r: r["val_miou"])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=130)
a1.plot(ep, [r["train_loss"] for r in h], label="train loss", color="#2a6fdb", lw=2)
a1.plot(ep, [r["val_loss"] for r in h], label="val loss", color="#e8590c", lw=2)
a1.set_xlabel("epoch")
a1.set_ylabel("CE + Dice loss")
a1.set_title("Loss")
a1.grid(alpha=0.3)
a1.legend(frameon=False)

a2.plot(ep, [r["val_miou"] for r in h], label="val mIoU (3 classes)", color="#2a6fdb", lw=2)
a2.plot(ep, [r["val_iou_lane"] for r in h], label="val IoU (lane)", color="#2f9e44", lw=2)
a2.axvline(best["epoch"], color="#888", ls="--", lw=1)
a2.annotate(f"best epoch {best['epoch']}\nmIoU {best['val_miou']:.3f}",
            (best["epoch"], best["val_miou"]), xytext=(-110, -40), textcoords="offset points",
            fontsize=9, color="#444")
a2.set_xlabel("epoch")
a2.set_ylabel("IoU")
a2.set_title("Validation IoU")
a2.set_ylim(0, 1)
a2.grid(alpha=0.3)
a2.legend(frameon=False, loc="lower right")

for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/loss_curve.png")
print("saved results/loss_curve.png")
