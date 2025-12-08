# efficientnetb0_lora.py
# ==========================================================
# EfficientNetB0 (ImageNet) - LoRA Adapter Head (PEFT)
# Backbone: freeze
# Trainable: LoRA A & B (+ optional bias head)
# ==========================================================

import os, json, time, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score

AUTOTUNE = tf.data.AUTOTUNE

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def require_split_dirs(data_dir: Path):
    ok = (data_dir/"train").exists() and (data_dir/"val").exists() and (data_dir/"test").exists()
    if not ok:
        raise ValueError("DATA_DIR harus punya train/val/test (contoh: dataset_sawit_split).")

def make_datasets(data_dir: Path, img_size=224, batch_size=32, seed=42):
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir/"train", image_size=(img_size, img_size),
        batch_size=batch_size, seed=seed, label_mode="int"
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir/"val", image_size=(img_size, img_size),
        batch_size=batch_size, seed=seed, label_mode="int", shuffle=False
    )
    test_ds = keras.utils.image_dataset_from_directory(
        data_dir/"test", image_size=(img_size, img_size),
        batch_size=batch_size, seed=seed, label_mode="int", shuffle=False
    )
    class_names = train_ds.class_names
    train_ds = train_ds.shuffle(1000, seed=seed).cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names

def plot_history(history, out_png: Path):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    ep = range(1, len(acc)+1)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(ep, acc, label="train")
    plt.plot(ep, val_acc, label="val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(ep, loss, label="train")
    plt.plot(ep, val_loss, label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def collect_preds(model, ds):
    y_true, y_pred, y_prob = [], [], []
    for bx, by in ds:
        prob = model.predict(bx, verbose=0)
        pred = np.argmax(prob, axis=1)
        y_true.append(by.numpy()); y_pred.append(pred); y_prob.append(prob)
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)

def plot_top_errors(images, y_true, y_pred, y_prob, class_names, out_png: Path, topk=12):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    conf = y_prob.max(axis=1)
    top = wrong[np.argsort(conf[wrong])[::-1]][:topk]
    cols = 4
    rows = int(np.ceil(len(top)/cols))
    plt.figure(figsize=(4*cols, 4*rows))
    for i, idx in enumerate(top, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(images[idx].astype(np.uint8))
        plt.axis("off")
        plt.title(f"T:{class_names[y_true[idx]]}\nP:{class_names[y_pred[idx]]} ({conf[idx]:.2f})", fontsize=10)
    plt.suptitle("Top Wrong Predictions (Most Confident)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

class LoRAHead(layers.Layer):
    """
    logits = Dense_base(x) + scale * (B(A(x)))
    Dense_base dibekukan (non-trainable), A dan B trainable (rank r).
    """
    def __init__(self, num_classes, r=8, alpha=16, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.r = r
        self.alpha = alpha
        self.scale = alpha / float(r)

        self.base = layers.Dense(num_classes, use_bias=True, trainable=False, name="base_head")
        self.A = layers.Dense(r, use_bias=False, name="lora_A")           # trainable
        self.B = layers.Dense(num_classes, use_bias=False, name="lora_B") # trainable

    def call(self, x):
        base_logits = self.base(x)
        delta = self.B(self.A(x)) * self.scale
        return base_logits + delta

def build_model(num_classes, img_size=224, dropout=0.3, r=8, alpha=16):
    inp = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Lambda(tf.keras.applications.efficientnet.preprocess_input)(inp)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)

    logits = LoRAHead(num_classes, r=r, alpha=alpha)(x)
    out = layers.Activation("softmax")(logits)

    model = keras.Model(inp, out, name="efficientnetb0_lora_head")
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--out_dir", default="out_efficientnetb0_lora", type=str)
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--r", default=8, type=int)
    ap.add_argument("--alpha", default=16, type=int)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    require_split_dirs(data_dir)

    out_dir = ensure_dir(Path(args.out_dir))
    figs = ensure_dir(out_dir/"figs")

    train_ds, val_ds, test_ds, class_names = make_datasets(
        data_dir, img_size=args.img_size, batch_size=args.batch_size, seed=args.seed
    )

    model = build_model(len(class_names), img_size=args.img_size, r=args.r, alpha=args.alpha)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    ckpt_path = out_dir/"best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    train_time = time.time() - t0

    plot_history(history, figs/"history_acc_loss.png")

    best = keras.models.load_model(ckpt_path, custom_objects={"LoRAHead": LoRAHead})
    test_loss, test_acc = best.evaluate(test_ds, verbose=0)

    y_true, y_pred, y_prob = collect_preds(best, test_ds)

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    (out_dir/"classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(values_format="d")
    plt.title("Confusion Matrix - EfficientNetB0 LoRA (Head)")
    plt.savefig(figs/"confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    imgs = []
    for bx, _ in test_ds:
        imgs.append(bx.numpy())
    imgs = np.concatenate(imgs, axis=0)
    plot_top_errors(imgs, y_true, y_pred, y_prob, class_names, figs/"top_errors.png", topk=12)

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")

    summary = {
        "mode": "LORA_HEAD",
        "model": "EfficientNetB0",
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.r,
        "lora_alpha": args.alpha,
        "classes": class_names,
        "train_time_sec": float(train_time),
        "test_loss": float(test_loss),
        "test_acc_eval": float(test_acc),
        "test_acc_sklearn": float(acc),
        "test_macro_f1": float(mf1),
        "trainable_params": int(np.sum([np.prod(v.shape) for v in best.trainable_weights])),
        "total_params": int(best.count_params()),
        "best_model_path": str(ckpt_path),
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[LoRA-HEAD] Test Acc={test_acc:.4f} | Macro-F1={mf1:.4f} | Saved -> {out_dir}")

if __name__ == "__main__":
    main()
