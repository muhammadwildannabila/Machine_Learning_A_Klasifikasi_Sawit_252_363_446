"""
Preprocessing Sawit - All-in-one (.py)
=====================================================
Langkah:
1) Import & Set Path
2) Koreksi Orientasi EXIF
3) Pengecilan Resolusi (opsional)
4) Normalisasi Warna (Gray World)
5) Letterbox Persegi + Resize 224x224
6) Simpan JPEG Quality 90
7) Verifikasi Visual (Before-After)

Struktur input yang didukung:
DATA_DIR/
  train/mentah|matang|busuk/*.jpg
  val/...
  test/...
atau:
DATA_DIR/
  mentah|matang|busuk/*.jpg

Output:
OUT_DIR/
  train/<class>/*.jpg
  val/<class>/*.jpg
  test/<class>/*.jpg
  (atau all/<class>/*.jpg)
+ outputs_preprocess/verify_before_after.png
"""

import os
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# -----------------------------
# Utils
# -----------------------------
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


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_class_folders(data_dir: Path):
    """
    Mendukung:
    - data_dir/train/<class>/*
    - data_dir/<class>/*
    """
    split_dirs = [data_dir / "train", data_dir / "val", data_dir / "test"]
    if any(d.exists() and d.is_dir() for d in split_dirs):
        roots = [d for d in split_dirs if d.exists() and d.is_dir()]
        class_names = set()
        for r in roots:
            for c in r.iterdir():
                if c.is_dir():
                    class_names.add(c.name)
        return roots, sorted(list(class_names)), True

    class_names = [d.name for d in data_dir.iterdir() if d.is_dir()]
    return [data_dir], sorted(class_names), False


def iter_images(roots, class_names, has_splits):
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


# -----------------------------
# Preprocess steps
# -----------------------------
def exif_correct(img: Image.Image) -> Image.Image:
    # koreksi rotasi berdasarkan EXIF
    return ImageOps.exif_transpose(img)


def downscale_if_needed(img: Image.Image, max_side: int) -> Image.Image:
    """
    Pengecilan resolusi (jika sisi terbesar > max_side).
    """
    if max_side is None or max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def gray_world(img: Image.Image) -> Image.Image:
    """
    Normalisasi warna Gray World:
    skala tiap channel agar mean R=G=B.
    """
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return img

    mean_rgb = arr.reshape(-1, 3).mean(axis=0)  # [meanR, meanG, meanB]
    mean_gray = mean_rgb.mean() + 1e-6

    scale = mean_gray / (mean_rgb + 1e-6)
    arr = arr * scale[None, None, :]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def letterbox_square(img: Image.Image, out_size: int, pad_color=(0, 0, 0)) -> Image.Image:
    """
    Letterbox jadi persegi: pad kanan/kiri/atas/bawah -> resize.
    """
    w, h = img.size
    side = max(w, h)
    new_img = Image.new("RGB", (side, side), pad_color)
    # center paste
    x = (side - w) // 2
    y = (side - h) // 2
    new_img.paste(img, (x, y))
    return new_img.resize((out_size, out_size), resample=Image.BICUBIC)


def preprocess_one(path: Path, max_side: int, out_size: int, grayworld: bool, pad_color=(0, 0, 0)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    # 2) EXIF
    img = exif_correct(img)
    # 3) downscale
    img = downscale_if_needed(img, max_side=max_side)
    # 4) gray world
    if grayworld:
        img = gray_world(img)
    # 5) letterbox + resize
    img = letterbox_square(img, out_size=out_size, pad_color=pad_color)
    return img


# -----------------------------
# Verification plot
# -----------------------------
def save_before_after_grid(pairs, out_png: Path, out_size: int):
    """
    pairs: list of (before_path, after_path)
    """
    n = len(pairs)
    cols = 2
    rows = n
    plt.figure(figsize=(8, 4 * rows))
    for i, (bp, ap) in enumerate(pairs, 1):
        b = Image.open(bp).convert("RGB")
        a = Image.open(ap).convert("RGB")

        plt.subplot(rows, cols, (i - 1) * cols + 1)
        plt.imshow(b)
        plt.axis("off")
        plt.title(f"Before\n{Path(bp).name}")

        plt.subplot(rows, cols, (i - 1) * cols + 2)
        plt.imshow(a.resize((out_size, out_size)))
        plt.axis("off")
        plt.title(f"After ({out_size}x{out_size})\n{Path(ap).name}")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path dataset split atau folder kelas langsung")
    parser.add_argument("--out_dir", type=str, required=True, help="Folder output hasil preprocessing")
    parser.add_argument("--out_size", type=int, default=224, help="Ukuran akhir (default 224)")
    parser.add_argument("--jpeg_quality", type=int, default=90, help="JPEG quality (default 90)")
    parser.add_argument("--max_side", type=int, default=1600, help="Downscale jika sisi terbesar > max_side (0 untuk disable)")
    parser.add_argument("--grayworld", action="store_true", help="Aktifkan Gray World color normalization")
    parser.add_argument("--pad_color", type=str, default="0,0,0", help="Warna padding letterbox 'R,G,B' (default 0,0,0)")
    parser.add_argument("--verify_n", type=int, default=6, help="Jumlah sampel verifikasi before-after")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite file output jika sudah ada")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR tidak ditemukan: {data_dir}")

    pad_color = tuple(int(x) for x in args.pad_color.split(","))
    if len(pad_color) != 3:
        raise ValueError("pad_color harus format 'R,G,B' misal 0,0,0")

    max_side = args.max_side if args.max_side and args.max_side > 0 else None

    roots, class_names, has_splits = find_class_folders(data_dir)
    if len(class_names) == 0:
        raise ValueError("Tidak menemukan folder kelas. Pastikan struktur dataset benar.")

    print("========================================")
    print("PREPROCESS START")
    print("DATA_DIR :", data_dir)
    print("OUT_DIR  :", out_dir)
    print("Classes  :", class_names)
    print("Has split:", has_splits)
    print("out_size :", args.out_size)
    print("max_side :", max_side)
    print("grayworld:", args.grayworld)
    print("jpeg_q   :", args.jpeg_quality)
    print("pad_color:", pad_color)
    print("========================================")

    # prepare output structure
    splits = [r.name for r in roots] if has_splits else ["all"]
    for sp in splits:
        for c in class_names:
            ensure_dir(out_dir / sp / c)

    # preprocess loop
    saved_pairs = []
    verify_pairs = []
    count = 0

    for split, cls, p in tqdm(list(iter_images(roots, class_names, has_splits)), desc="Preprocessing"):
        rel_name = p.stem + ".jpg"  # simpan sebagai jpg
        out_path = out_dir / split / cls / rel_name

        if out_path.exists() and not args.overwrite:
            continue

        try:
            img_out = preprocess_one(
                p,
                max_side=max_side,
                out_size=args.out_size,
                grayworld=args.grayworld,
                pad_color=pad_color
            )
            img_out.save(out_path, format="JPEG", quality=args.jpeg_quality, optimize=True)

            count += 1

            # simpan pair untuk verifikasi
            if len(verify_pairs) < args.verify_n:
                verify_pairs.append((str(p), str(out_path)))

        except Exception as e:
            print(f"[WARN] Skip {p} -> {e}")
            continue

    # verifikasi before-after
    verify_dir = ensure_dir(out_dir / "outputs_preprocess")
    if len(verify_pairs) > 0:
        out_png = verify_dir / "verify_before_after.png"
        save_before_after_grid(verify_pairs, out_png, out_size=args.out_size)
        print(f"[OK] Verifikasi saved -> {out_png}")
    else:
        print("[WARN] Tidak ada sample verifikasi (mungkin semua sudah ada dan overwrite=false).")

    print("========================================")
    print(f"PREPROCESS DONE ✅ | saved/updated files: {count}")
    print("Output folder:", out_dir)
    print("========================================")


if __name__ == "__main__":
    mount_drive_if_colab()
    main()
