# color_xgboost.py
# ============================================
# Feature Extraction (COLOR) + XGBoost
# ============================================

import argparse, time, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

try:
    import cv2
except Exception:
    cv2 = None

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def has_split_dirs(data_dir: Path):
    return (data_dir/"train").exists() and (data_dir/"test").exists()

def list_images(root: Path):
    items = []
    for cls_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        cls = cls_dir.name
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append((str(p), cls))
    return items

def color_features_pil(img: Image.Image, bins=16):
    # fallback tanpa cv2: RGB mean/std + RGB hist (coarse)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    feats = []
    feats.extend(arr.reshape(-1,3).mean(axis=0).tolist())
    feats.extend(arr.reshape(-1,3).std(axis=0).tolist())

    # coarse hist per channel
    for c in range(3):
        h, _ = np.histogram(arr[:,:,c], bins=bins, range=(0,1), density=True)
        feats.extend(h.tolist())
    return np.array(feats, dtype=np.float32)

def color_features_cv2(img: Image.Image, bins=16):
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    rgb = arr.astype(np.float32) / 255.0

    # RGB mean/std
    feats = []
    feats.extend(rgb.reshape(-1,3).mean(axis=0).tolist())
    feats.extend(rgb.reshape(-1,3).std(axis=0).tolist())

    # HSV mean/std
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,0] = hsv[:,:,0] / 179.0  # H norm 0..1
    hsv[:,:,1] = hsv[:,:,1] / 255.0
    hsv[:,:,2] = hsv[:,:,2] / 255.0
    feats.extend(hsv.reshape(-1,3).mean(axis=0).tolist())
    feats.extend(hsv.reshape(-1,3).std(axis=0).tolist())

    # HSV hist (H,S,V) bins=16 each
    for ch in range(3):
        h, _ = np.histogram(hsv[:,:,ch], bins=bins, range=(0,1), density=True)
        feats.extend(h.tolist())

    return np.array(feats, dtype=np.float32)

def extract_Xy(items, img_size=224, bins=16):
    X, y, paths = [], [], []
    for path, cls in items:
        try:
            img = Image.open(path).convert("RGB").resize((img_size, img_size))
            if cv2 is not None:
                feat = color_features_cv2(img, bins=bins)
            else:
                feat = color_features_pil(img, bins=bins)
            X.append(feat)
            y.append(cls)
            paths.append(path)
        except Exception:
            continue
    return np.array(X), np.array(y), np.array(paths)

def plot_cv_results(grid, out_png: Path, title: str):
    r = grid.cv_results_
    mean = r["mean_test_score"]; std = r["std_test_score"]; params = r["params"]
    order = np.argsort(mean)[::-1][:10]
    labels = [str(params[i]) for i in order]
    scores = mean[order]; errs = std[order]
    plt.figure(figsize=(10,5))
    plt.bar(range(len(scores)), scores, yerr=errs)
    plt.xticks(range(len(scores)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("CV score (macro-F1)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def plot_top_errors(proba, y_true, y_pred, paths, classes, out_png: Path, topk=12, img_size=224):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    conf = proba.max(axis=1)
    top = wrong[np.argsort(conf[wrong])[::-1]][:topk]

    cols = 4
    rows = int(np.ceil(len(top)/cols))
    plt.figure(figsize=(4*cols, 4*rows))
    for i, idx in enumerate(top, 1):
        img = Image.open(paths[idx]).convert("RGB").resize((img_size, img_size))
        plt.subplot(rows, cols, i)
        plt.imshow(img); plt.axis("off")
        plt.title(f"T:{classes[y_true[idx]]}\nP:{classes[y_pred[idx]]} ({conf[idx]:.2f})", fontsize=10)
    plt.suptitle("Top Wrong Predictions (Most Confident)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out_color_xgb")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    ensure_dir(out_dir/"figs")

    if has_split_dirs(data_dir):
        train_items = list_images(data_dir/"train")
        val_items   = list_images(data_dir/"val") if (data_dir/"val").exists() else []
        test_items  = list_images(data_dir/"test")
    else:
        all_items = list_images(data_dir)
        train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=args.seed, stratify=[c for _,c in all_items])
        val_items = []

    X_train, y_train, _ = extract_Xy(train_items, img_size=args.img_size, bins=args.bins)
    X_val, y_val, _     = extract_Xy(val_items, img_size=args.img_size, bins=args.bins) if len(val_items) else (None,None,None)
    X_test, y_test, p_test = extract_Xy(test_items, img_size=args.img_size, bins=args.bins)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    classes = le.classes_

    if X_val is not None and len(X_val) > 0:
        y_val_enc = le.transform(y_val)
        X_fit = np.concatenate([X_train, X_val], axis=0)
        y_fit = np.concatenate([y_train_enc, y_val_enc], axis=0)
    else:
        X_fit, y_fit = X_train, y_train_enc

    print("Feature dim:", X_train.shape[1], "| Classes:", list(classes))

    try:
        from xgboost import XGBClassifier
    except Exception:
        raise RuntimeError("Install xgboost: pip install xgboost")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=args.seed
    )

    param_grid = {
        "n_estimators": [300, 600],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    grid = GridSearchCV(model, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)

    t0 = time.time()
    grid.fit(X_train, y_train_enc)
    print("Best params:", grid.best_params_)
    print("Best CV macro-F1:", grid.best_score_, "| time:", round(time.time()-t0,2),"s")

    plot_cv_results(grid, out_dir/"figs"/"cv_top10.png", "COLOR + XGBoost CV Top-10 (macro-F1)")

    best = grid.best_estimator_
    best.fit(X_fit, y_fit)

    t0 = time.time()
    proba = best.predict_proba(X_test)
    y_pred = proba.argmax(axis=1)
    infer_time = time.time()-t0

    acc = accuracy_score(y_test_enc, y_pred)
    mf1 = f1_score(y_test_enc, y_pred, average="macro")

    report = classification_report(y_test_enc, y_pred, target_names=classes, zero_division=0)
    (out_dir/"classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test_enc, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(values_format="d")
    plt.title("Confusion Matrix - COLOR + XGBoost")
    plt.savefig(out_dir/"figs"/"confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    plot_top_errors(proba, y_test_enc, y_pred, p_test, classes, out_dir/"figs"/"top_errors.png")

    summary = {
        "feature": "COLOR",
        "model": "XGBoost",
        "img_size": args.img_size,
        "bins": args.bins,
        "feature_dim": int(X_train.shape[1]),
        "best_params": grid.best_params_,
        "best_cv_macro_f1": float(grid.best_score_),
        "test_acc": float(acc),
        "test_macro_f1": float(mf1),
        "inference_time_sec": float(infer_time),
        "classes": classes.tolist(),
        "cv2_used": bool(cv2 is not None),
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Test Acc: {acc:.4f} | Macro-F1: {mf1:.4f} | Saved -> {out_dir}")

if __name__ == "__main__":
    main()
