# vit_b16_freeze.py
# ==========================================================
# ViT-B/16 (timm) - FREEZE backbone (train head only)
# Output:
# - figs/history_acc_loss.png
# - figs/confusion_matrix.png
# - classification_report.txt
# - figs/top_errors.png
# - best_model.pth
# - summary.json
# ==========================================================

import argparse, json, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import timm

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def require_split_dirs(data_dir: Path):
    for s in ["train","val","test"]:
        if not (data_dir/s).exists():
            raise ValueError("DATA_DIR harus punya train/val/test.")

def get_loaders(data_dir: Path, img_size=224, batch_size=32, num_workers=2):
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    train_ds = datasets.ImageFolder(data_dir/"train", transform=tfm_train)
    val_ds   = datasets.ImageFolder(data_dir/"val",   transform=tfm_eval)
    test_ds  = datasets.ImageFolder(data_dir/"test",  transform=tfm_eval)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl

def plot_history(hist, out_png: Path):
    ep = range(1, len(hist["train_acc"])+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(ep, hist["train_acc"], label="train")
    plt.plot(ep, hist["val_acc"], label="val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(ep, hist["train_loss"], label="train")
    plt.plot(ep, hist["val_loss"], label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    all_y, all_p, all_prob = [], [], []
    total_loss, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_y.append(y.cpu().numpy())
        all_p.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    y_prob = np.concatenate(all_prob)
    return total_loss/n, (y_true==y_pred).mean(), y_true, y_pred, y_prob

def train_one_epoch(model, dl, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss/n, correct/n

def plot_top_errors(model, test_ds, test_dl, device, class_names, out_png: Path, topk=12):
    model.eval()
    wrong = []
    with torch.no_grad():
        idx_global = 0
        for x, y in test_dl:
            x = x.to(device)
            prob = torch.softmax(model(x), dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)
            conf = prob.max(axis=1)
            y_np = y.numpy()
            for i in range(len(y_np)):
                if pred[i] != y_np[i]:
                    wrong.append((idx_global+i, conf[i], pred[i], y_np[i]))
            idx_global += len(y_np)

    if len(wrong) == 0:
        return
    wrong.sort(key=lambda t: t[1], reverse=True)
    pick = wrong[:topk]

    from PIL import Image
    cols = 4
    rows = int(np.ceil(len(pick)/cols))
    plt.figure(figsize=(4*cols, 4*rows))
    for i, (idx, conf, p, t) in enumerate(pick, 1):
        img_path, _ = test_ds.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((224,224))
        plt.subplot(rows, cols, i)
        plt.imshow(img); plt.axis("off")
        plt.title(f"T:{class_names[t]}\nP:{class_names[p]} ({conf:.2f})", fontsize=10)
    plt.suptitle("Top Wrong Predictions (Most Confident)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--out_dir", default="out_vit_b16_freeze", type=str)
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--patience", default=6, type=int)
    ap.add_argument("--num_workers", default=2, type=int)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    require_split_dirs(data_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    figs = ensure_dir(out_dir/"figs")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = get_loaders(
        data_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    class_names = train_ds.classes
    num_classes = len(class_names)

    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    freeze_backbone(model)
    model.to(device)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    best_val, bad = -1, 0
    best_path = out_dir/"best_model.pth"
    hist = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    t0 = time.time()
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, device)
        va_loss, va_acc, _, _, _ = evaluate(model, val_dl, device)

        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss);   hist["val_acc"].append(va_acc)

        print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), best_path)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    train_time = time.time() - t0
    plot_history(hist, figs/"history_acc_loss.png")

    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc, y_true, y_pred, y_prob = evaluate(model, test_dl, device)

    (out_dir/"classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
        encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(values_format="d")
    plt.title("Confusion Matrix - ViT-B/16 Freeze")
    plt.savefig(figs/"confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    plot_top_errors(model, test_ds, test_dl, device, class_names, figs/"top_errors.png", topk=12)

    acc_sk = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")

    summary = {
        "mode":"FREEZE",
        "model":"ViT-B/16 (timm vit_base_patch16_224)",
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs_ran": len(hist["train_acc"]),
        "lr": args.lr,
        "classes": class_names,
        "train_time_sec": float(train_time),
        "best_val_acc": float(best_val),
        "test_loss": float(te_loss),
        "test_acc_eval": float(te_acc),
        "test_acc_sklearn": float(acc_sk),
        "test_macro_f1": float(mf1),
        "best_model_path": str(best_path),
        "device": device,
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[ViT-B/16 FREEZE] Test Acc={te_acc:.4f} | Macro-F1={mf1:.4f} | Saved -> {out_dir}")

if __name__ == "__main__":
    main()
