import os
import shutil
from sklearn.model_selection import train_test_split

# GANTI sesuai lokasi dataset awalmu
BASE_DIR = "/content/drive/MyDrive/dataset_sawit"   
OUT_DIR  = "/content/drive/MyDrive/dataset_sawit_split"

classes = ["mentah", "matang", "busuk"]
os.makedirs(OUT_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

# Seed biar reproducible
RANDOM_STATE = 42

for cls in classes:
    cls_dir = os.path.join(BASE_DIR, cls)
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # 70% train, 30% sisa
    train_imgs, temp_imgs = train_test_split(
        images, test_size=0.30, random_state=RANDOM_STATE
    )
    # dari 30% → 15% val, 15% test (jadi 0.5:0.5 dari sisa)
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.50, random_state=RANDOM_STATE
    )

    def copy_files(file_list, split_name):
        for fname in file_list:
            src = os.path.join(cls_dir, fname)
            dst = os.path.join(OUT_DIR, split_name, cls, fname)
            shutil.copy2(src, dst)

    copy_files(train_imgs, "train")
    copy_files(val_imgs, "val")
    copy_files(test_imgs, "test")

    print(f"{cls}: total={len(images)}, train={len(train_imgs)}, "
          f"val={len(val_imgs)}, test={len(test_imgs)}")
