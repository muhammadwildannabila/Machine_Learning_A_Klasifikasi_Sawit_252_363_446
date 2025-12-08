"""
EDA Sawit - All-in-one (.py)
=====================================================
Cocok untuk Colab / Local.

Struktur dataset yang didukung:
DATA_DIR/
  train/mentah|matang|busuk/*.jpg
  val/...
  test/...
atau:
DATA_DIR/
  mentah|matang|busuk/*.jpg

Output:
- outputs/eda_metadata.csv
- outputs/*.png (plot)
- outputs/duplicates_exact.csv
- outputs/duplicates_phash.csv (kalau imagehash tersedia)
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

# Optional deps
try:
    import cv2
except Exception:
    cv2 = None

try:
    import imagehash
except Exception:
    imagehash = None


# -----------------------------
# Helpers
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def mount_drive_if_colab():
    if is_colab():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("[OK] Google Drive mounted at /content/drive")
        except Exception as e:
            print("[WARN] Gagal mount drive:", e)


def find_class_folders(data_dir: Path):
    """
    Mendukung:
    - data_dir/train/<class>/*
    - data_dir/<class>/*
    """
    # Case A: has train/val/test
    split_dirs = [data_dir / "train", data_dir / "val", data_dir / "test"]
    if any(d.exists() and d.is_dir() for d in split_dirs):
        # choose existing split dirs
        roots = [d for d in split_dirs if d.exists() and d.is_dir()]
        class_names = set()
        for r in roots:
            for c in r.iterdir():
                if c.is_dir():
                    class_names.add(c.name)
        return roots, sorted(list(class_names)), True

    # Case B: direct classes
    class_names = [d.name for d in data_dir.iterdir() if d.is_dir()]
    return [data_dir], sorted(class_names), False


def iter_images(roots, class_names, has_splits):
    """
    yield: (split, class, filepath)
    """
    if has_splits:
        for root in roots:
            split = root.name
            for c in class_names:
                cdir = root / c
                if not cdir.exists():
                    continue
                for p in cdir.rglob("*"):
                    if p.is_file() and p.suffix.lower() in IMG_EXTS:
                        yield split, c, p
    else:
        split = "all"
        for c in class_names:
            cdir = roots[0] / c
            if not cdir.exists():
                continue
            for p in cdir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    yield split, c, p


def safe_open_image(path: Path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


def compute_brightness_rgb(img_np):
    # img_np: uint8 RGB
    # brightness = mean grayscale
    gray = (0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2])
    return float(gray.mean())


def compute_blur_variance_laplacian(img_np):
    # Blur metric: var(Laplacian(gray)) (lebih kecil = lebih blur)
    if cv2 is None:
        return np.nan
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_hue_mean(img_np):
    # Hue mean (0..180 OpenCV) -> kita normalisasi 0..1
    if cv2 is None:
        return np.nan
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.float32) / 180.0
    return float(h.mean())


def compute_lab_ab(img_np):
    # Lab a,b mean (OpenCV)
    if cv2 is None:
        return (np.nan, np.nan)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[:, :, 1].mean()
    b = lab[:, :, 2].mean()
    return float(a), float(b)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# Plotting
# -----------------------------
def plot_class_distribution(df, outdir: Path):
    plt.figure(figsize=(7, 4))
    counts = df["class"].value_counts().sort_index()
    plt.bar(counts.index, counts.values)
    plt.title("Distribusi Jumlah Citra per Kelas")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah")
    savefig(outdir / "01_class_distribution.png")


def plot_sample_grid(df, outdir: Path, per_class=8, seed=42):
    rng = np.random.default_rng(seed)
    classes = sorted(df["class"].unique().tolist())

    for c in classes:
        sdf = df[df["class"] == c].copy()
        if len(sdf) == 0:
            continue
        n = min(per_class, len(sdf))
        idx = rng.choice(sdf.index.to_numpy(), size=n, replace=False)
        samples = sdf.loc[idx, "path"].tolist()

        cols = 4
        rows = int(np.ceil(n / cols))
        plt.figure(figsize=(4 * cols, 4 * rows))
        for i, p in enumerate(samples, 1):
            img = safe_open_image(Path(p))
            if img is None:
                continue
            plt.subplot(rows, cols, i)
            plt.imshow(img)
            plt.axis("off")
        plt.suptitle(f"Grid Sampel - {c}", y=1.02, fontsize=14)
        savefig(outdir / f"02_sample_grid_{c}.png")


def plot_resolution_aspect(df, outdir: Path):
    # scatter width vs height
    plt.figure(figsize=(6, 5))
    for c in sorted(df["class"].unique()):
        sdf = df[df["class"] == c]
        plt.scatter(sdf["width"], sdf["height"], s=8, alpha=0.5, label=c)
    plt.title("Resolusi Citra (Width vs Height)")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.legend()
    savefig(outdir / "03_resolution_scatter.png")

    # aspect ratio histogram
    plt.figure(figsize=(7, 4))
    for c in sorted(df["class"].unique()):
        sdf = df[df["class"] == c]
        plt.hist(sdf["aspect_ratio"].dropna(), bins=40, alpha=0.5, label=c)
    plt.title("Distribusi Rasio Aspek (W/H)")
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Count")
    plt.legend()
    savefig(outdir / "04_aspect_ratio_hist.png")


def plot_brightness(df, outdir: Path):
    plt.figure(figsize=(7, 4))
    for c in sorted(df["class"].unique()):
        sdf = df[df["class"] == c]
        plt.hist(sdf["brightness"].dropna(), bins=40, alpha=0.5, label=c)
    plt.title("Distribusi Kecerahan (Exposure)")
    plt.xlabel("Brightness (mean grayscale)")
    plt.ylabel("Count")
    plt.legend()
    savefig(outdir / "05_brightness_hist.png")


def plot_blur(df, outdir: Path):
    plt.figure(figsize=(7, 4))
    for c in sorted(df["class"].unique()):
        sdf = df[df["class"] == c]
        vals = sdf["blur_var"].dropna()
        if len(vals) == 0:
            continue
        plt.hist(vals, bins=40, alpha=0.5, label=c)
    plt.title("Distribusi Blur (Var Laplacian) — kecil = lebih blur")
    plt.xlabel("Blur Variance")
    plt.ylabel("Count")
    plt.legend()
    savefig(outdir / "06_blur_hist.png")


def plot_file_size(df, outdir: Path):
    plt.figure(figsize=(7, 4))
    for c in sorted(df["class"].unique()):
        sdf = df[df["class"] == c]
        plt.hist(sdf["filesize_kb"].dropna(), bins=40, alpha=0.5, label=c)
    plt.title("Distribusi Ukuran Berkas per Kelas")
    plt.xlabel("File size (KB)")
    plt.ylabel("Count")
    plt.legend()
    savefig(outdir / "07_filesize_hist.png")


def plot_hue_density(df, outdir: Path):
    plt.figure(figsize=(7, 4))
    for c in sorted(df["class"].unique()):
        sdf = df[df["class"] == c]
        vals = sdf["hue_mean"].dropna()
        if len(vals) == 0:
            continue
        # "density" pakai hist normalized (tanpa seaborn)
        hist, edges = np.histogram(vals, bins=60, range=(0, 1), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        plt.plot(centers, hist, label=c)
    plt.title("Analisis Warna: Hue Density (0..1)")
    plt.xlabel("Hue (normalized)")
    plt.ylabel("Density")
    plt.legend()
    savefig(outdir / "08_hue_density.png")


def plot_lab_scatter(df, outdir: Path):
    if df["lab_a"].isna().all() or df["lab_b"].isna().all():
        return
    plt.figure(figsize=(6, 5))
    for c in sorted(df["class"].unique()):
        sdf = df[df["class"] == c]
        plt.scatter(sdf["lab_a"], sdf["lab_b"], s=8, alpha=0.5, label=c)
    plt.title("Analisis Warna: Lab Scatter (a vs b)")
    plt.xlabel("Lab a*")
    plt.ylabel("Lab b*")
    plt.legend()
    savefig(outdir / "09_lab_scatter.png")


# -----------------------------
# Duplicate detection
# -----------------------------
def md5_of_file(path: Path, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def detect_exact_duplicates(df, outdir: Path):
    # exact duplicates by MD5 (file content)
    groups = defaultdict(list)
    for p in tqdm(df["path"].tolist(), desc="MD5 hashing"):
        try:
            groups[md5_of_file(Path(p))].append(p)
        except Exception:
            continue

    dup_rows = []
    for k, paths in groups.items():
        if len(paths) > 1:
            base = paths[0]
            for other in paths[1:]:
                dup_rows.append({"hash_md5": k, "base": base, "dup": other})

    dup_df = pd.DataFrame(dup_rows)
    dup_df.to_csv(outdir / "duplicates_exact.csv", index=False)
    print(f"[OK] Exact duplicates: {len(dup_df)} pairs -> outputs/duplicates_exact.csv")


def detect_phash_duplicates(df, outdir: Path, max_images=3000, thresh=6):
    # near-duplicates by perceptual hash
    if imagehash is None:
        print("[SKIP] imagehash belum terpasang. (pip install ImageHash)")
        return

    paths = df["path"].tolist()
    if len(paths) > max_images:
        paths = paths[:max_images]
        print(f"[INFO] phash dibatasi max_images={max_images}")

    hashes = []
    for p in tqdm(paths, desc="pHash hashing"):
        img = safe_open_image(Path(p))
        if img is None:
            continue
        try:
            ph = imagehash.phash(img.resize((224, 224)))
            hashes.append((p, ph))
        except Exception:
            continue

    # brute force grouping by distance threshold (cukup untuk skala ribuan)
    dup_rows = []
    for i in tqdm(range(len(hashes)), desc="pHash compare"):
        pi, hi = hashes[i]
        for j in range(i + 1, len(hashes)):
            pj, hj = hashes[j]
            if hi - hj <= thresh:
                dup_rows.append({"path_a": pi, "path_b": pj, "phash_a": str(hi), "phash_b": str(hj), "dist": int(hi - hj)})

    dup_df = pd.DataFrame(dup_rows).sort_values("dist") if len(dup_rows) else pd.DataFrame(columns=["path_a","path_b","phash_a","phash_b","dist"])
    dup_df.to_csv(outdir / "duplicates_phash.csv", index=False)
    print(f"[OK] Near duplicates (pHash<= {thresh}): {len(dup_df)} pairs -> outputs/duplicates_phash.csv")


# -----------------------------
# Main EDA
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path dataset split (train/val/test) atau folder kelas langsung")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Folder output EDA")
    parser.add_argument("--max_images", type=int, default=5000, help="Batasi jumlah gambar yang dianalisis (untuk cepat)")
    parser.add_argument("--sample_per_class", type=int, default=8, help="Jumlah sampel grid per kelas")
    parser.add_argument("--phash_max_images", type=int, default=2000, help="Batas gambar untuk pHash (biar tidak lama)")
    parser.add_argument("--phash_thresh", type=int, default=6, help="Threshold pHash distance untuk near-duplicate")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR tidak ditemukan: {data_dir}")

    print("========================================")
    print("EDA START")
    print("DATA_DIR:", data_dir)
    print("OUT_DIR :", out_dir)
    print("OpenCV  :", "OK" if cv2 is not None else "MISSING (blur/hue/lab jadi NaN)")
    print("ImageHash:", "OK" if imagehash is not None else "MISSING (near-duplicate skip)")
    print("========================================")

    roots, class_names, has_splits = find_class_folders(data_dir)
    if len(class_names) == 0:
        raise ValueError("Tidak menemukan folder kelas. Pastikan struktur dataset benar.")

    # 1) Build metadata DF
    rows = []
    it = list(iter_images(roots, class_names, has_splits))
    if len(it) > args.max_images:
        it = it[:args.max_images]
        print(f"[INFO] metadata dibatasi max_images={args.max_images}")

    for split, cls, p in tqdm(it, desc="Build metadata"):
        img = safe_open_image(p)
        if img is None:
            continue
        w, h = img.size
        aspect = w / h if h else np.nan
        fkb = p.stat().st_size / 1024.0

        img_np = np.array(img, dtype=np.uint8)
        bright = compute_brightness_rgb(img_np)
        blur_v = compute_blur_variance_laplacian(img_np)
        hue_m = compute_hue_mean(img_np)
        lab_a, lab_b = compute_lab_ab(img_np)

        rows.append({
            "split": split,
            "class": cls,
            "path": str(p),
            "filename": p.name,
            "width": w,
            "height": h,
            "aspect_ratio": aspect,
            "filesize_kb": fkb,
            "brightness": bright,
            "blur_var": blur_v,
            "hue_mean": hue_m,
            "lab_a": lab_a,
            "lab_b": lab_b,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "eda_metadata.csv", index=False)
    print(f"[OK] Saved metadata -> {out_dir / 'eda_metadata.csv'} | rows={len(df)}")

    # 2) Distribusi kelas
    plot_class_distribution(df, out_dir)

    # 3) Grid sampel per kelas
    plot_sample_grid(df, out_dir, per_class=args.sample_per_class, seed=42)

    # 4-5) Resolusi & rasio aspek
    plot_resolution_aspect(df, out_dir)

    # 6) Exposure
    plot_brightness(df, out_dir)

    # 7) Blur
    plot_blur(df, out_dir)

    # 8) Ukuran berkas
    plot_file_size(df, out_dir)

    # 9) Hue density
    plot_hue_density(df, out_dir)

    # 10) Lab scatter
    plot_lab_scatter(df, out_dir)

    # 11) Duplicate detection
    detect_exact_duplicates(df, out_dir)
    detect_phash_duplicates(df, out_dir, max_images=args.phash_max_images, thresh=args.phash_thresh)

    print("========================================")
    print("EDA DONE ✅")
    print("Check folder:", out_dir)
    print("========================================")


if __name__ == "__main__":
    # (Optional) mount drive otomatis kalau di Colab
    mount_drive_if_colab()
    main()
