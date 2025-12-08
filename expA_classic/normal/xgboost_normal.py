# classic_xgboost_normal.py
# ============================================
# Classic Normal (Pixel Flatten) - XGBoost
# Output: CV plot, CM, report, error analysis
# ============================================

import os, argparse, time, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

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

def load_xy(items, img_size=64, max_images=None, seed=42):
    if max_images and len(items) > max_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(items), size=max_images, replace=False)
        items = [items[i] for i in idx]

    X, y, paths = [], [], []
    for path, cls in items:
        try:
            img = Image.open(path).convert("RGB").resize((img_size, img_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X.append(arr.reshape(-1))
            y.append(cls)
            paths.append(path)
        except Exception:
            continue
    return np.array(X), np.array(y), np.array(paths)

def plot_cv_results(grid, out_png: Path, title: str):
    results = grid.cv_results_
    mean = results["mean_test_score"]
    std = results["std_test_score"]
    params = results["params"]

    order = np.argsort(mean)[::-1][:10]
    labels = [str(params[i]) for i in order]
    scores = mean[order]
    errs = std[order]

    plt.figure(figsize=(10, 5))
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
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"T:{classes[y_true[idx]]}\nP:{classes[y_pred[idx]]} ({conf[idx]:.2f})", fontsize=10)
    plt.suptitle("Top Wrong Predictions (Most Confident)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs_classic_xgb")
    parser.add_argument("--img_size", type=int, default=64, help="resize for pixel flatten (64 recommended)")
    parser.add_argument("--max_images", type=int, default=0, help="limit images (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    (out_dir/"figs").mkdir(exist_ok=True)

    # ----- Load dataset -----
    if has_split_dirs(data_dir):
        train_items = list_images(data_dir/"train")
        val_items   = list_images(data_dir/"val") if (data_dir/"val").exists() else []
        test_items  = list_images(data_dir/"test")
    else:
        all_items = list_images(data_dir)
        train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=args.seed, stratify=[c for _,c in all_items])
        val_items = []

    X_train, y_train, _ = load_xy(train_items, img_size=args.img_size, max_images=args.max_images or None, seed=args.seed)
    X_val, y_val, _     = load_xy(val_items,   img_size=args.img_size, max_images=args.max_images or None, seed=args.seed) if len(val_items) else (None,None,None)
    X_test, y_test, p_test = load_xy(test_items, img_size=args.img_size, max_images=None, seed=args.seed)

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

    print("Classes:", list(classes))
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # ----- Model + GridSearch -----
    try:
        from xgboost import XGBClassifier
    except Exception:
        raise RuntimeError("Install xgboost dulu: pip install xgboost")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=args.seed
    )

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    grid = GridSearchCV(
        model, param_grid=param_grid,
        scoring="f1_macro",
        cv=cv, n_jobs=-1, verbose=1
    )

    t0 = time.time()
    grid.fit(X_train, y_train_enc)
    print("GridSearch time:", round(time.time()-t0, 2), "sec")
    print("Best params:", grid.best_params_)
    print("Best CV macro-F1:", grid.best_score_)

    plot_cv_results(grid, out_dir/"figs"/"cv_top10.png", "XGBoost CV Top-10 (macro-F1)")

    best = grid.best_estimator_
    best.fit(X_fit, y_fit)

    # ----- Test evaluation -----
    t0 = time.time()
    proba = best.predict_proba(X_test)
    y_pred = proba.argmax(axis=1)
    infer_time = time.time() - t0

    acc = accuracy_score(y_test_enc, y_pred)
    mf1 = f1_score(y_test_enc, y_pred, average="macro")

    print(f"Test Acc: {acc:.4f} | Macro-F1: {mf1:.4f} | Inference time: {infer_time:.2f}s")

    report = classification_report(y_test_enc, y_pred, target_names=classes, zero_division=0)
    (out_dir/"classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test_enc, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(values_format="d")
    plt.title("Confusion Matrix - XGBoost (Pixel Flatten)")
    plt.savefig(out_dir/"figs"/"confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    plot_top_errors(proba, y_test_enc, y_pred, p_test, classes, out_dir/"figs"/"top_errors.png", topk=12)

    summary = {
        "model": "XGBoost_pixel_flatten",
        "img_size": args.img_size,
        "best_params": grid.best_params_,
        "best_cv_macro_f1": float(grid.best_score_),
        "test_acc": float(acc),
        "test_macro_f1": float(mf1),
        "inference_time_sec": float(infer_time),
        "classes": classes.tolist(),
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
