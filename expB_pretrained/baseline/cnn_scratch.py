# cnn_scratch.py
# ==========================================================
# CNN Scratch (from-scratch) untuk Klasifikasi Sawit
# Output:
# - history_acc_loss.png
# - confusion_matrix.png
# - classification_report.txt
# - top_errors.png
# - best_model.keras
# - summary.json
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
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def has_split_dirs(data_dir: Path):
    return (data_dir/"train").exists() and (data_dir/"val").exists() and (data_dir/"test").exists()

def get_class_names(train_dir: Path):
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if len(class_names) == 0:
        raise ValueError("Tidak menemukan folder kelas di train_dir.")
    return class_names

def build_cnn_scratch(num_classes: int, img_size=224):
    inputs = keras.Input(shape=(img_size, img_size, 3))

    x = layers.Rescaling(1./255)(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="cnn_scratch")
    return model

def make_datasets(data_dir: Path, img_size=224, batch_size=32, seed=42):
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"
    test_dir  = data_dir / "test"

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        seed=seed,
        label_mode="int"
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        seed=seed,
        label_mode="int",
        shuffle=False
    )

    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        seed=seed,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names

    # cache & prefetch
    train_ds = train_ds.shuffle(1000, seed=seed).cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

def plot_history(history, out_png: Path):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label="train")
    plt.plot(epochs, val_acc, label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def collect_preds(model, ds):
    y_true = []
    y_pred = []
    y_prob = []
    for batch_x, batch_y in ds:
        prob = model.predict(batch_x, verbose=0)
        pred = np.argmax(prob, axis=1)
        y_true.append(batch_y.numpy())
        y_pred.append(pred)
        y_prob.append(prob)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    return y_true, y_pred, y_prob

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
        plt.imshow(images[idx])
        plt.axis("off")
        plt.title(
            f"T:{class_names[y_true[idx]]}\nP:{class_names[y_pred[idx]]} ({conf[idx]:.2f})",
            fontsize=10
        )

    plt.suptitle("Top Wrong Predictions (Most Confident)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="folder yang ada train/val/test")
    parser.add_argument("--out_dir", type=str, default="out_cnn_scratch")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not has_split_dirs(data_dir):
        raise ValueError("DATA_DIR harus punya train/val/test. (contoh: dataset_sawit_split)")

    out_dir = ensure_dir(Path(args.out_dir))
    figs = ensure_dir(out_dir/"figs")

    print("====================================")
    print("CNN SCRATCH START")
    print("DATA_DIR:", data_dir)
    print("OUT_DIR :", out_dir)
    print("img_size:", args.img_size, "| batch:", args.batch_size, "| epochs:", args.epochs)
    print("====================================")

    # Reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    train_ds, val_ds, test_ds, class_names = make_datasets(
        data_dir, img_size=args.img_size, batch_size=args.batch_size, seed=args.seed
    )

    model = build_cnn_scratch(num_classes=len(class_names), img_size=args.img_size)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    ckpt_path = out_dir / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        )
    ]

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    train_time = time.time() - t0

    # save history plot
    plot_history(history, figs/"history_acc_loss.png")

    # load best model
    best_model = keras.models.load_model(ckpt_path)

    # evaluate on test
    t0 = time.time()
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    infer_time = time.time() - t0

    # collect predictions
    y_true, y_pred, y_prob = collect_preds(best_model, test_ds)

    # classification report
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    (out_dir/"classification_report.txt").write_text(report, encoding="utf-8")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(values_format="d")
    plt.title("Confusion Matrix - CNN Scratch")
    plt.savefig(figs/"confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    # error analysis: ambil sample image dari test_ds untuk grid
    test_images = []
    test_labels = []
    for bx, by in test_ds:
        test_images.append(bx.numpy().astype(np.uint8))
        test_labels.append(by.numpy())
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    plot_top_errors(test_images, y_true, y_pred, y_prob, class_names, figs/"top_errors.png", topk=12)

    # summary
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")

    summary = {
        "model": "CNN_Scratch",
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "classes": class_names,
        "train_time_sec": float(train_time),
        "test_loss": float(test_loss),
        "test_acc_eval": float(test_acc),
        "test_acc_sklearn": float(acc),
        "test_macro_f1": float(mf1),
        "inference_time_sec": float(infer_time),
        "best_model_path": str(ckpt_path)
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("====================================")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Macro-F1: {mf1:.4f}")
    print("Saved to:", out_dir)
    print("====================================")

if __name__ == "__main__":
    main()
