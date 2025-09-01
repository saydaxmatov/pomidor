# -*- coding: utf-8 -*-
"""
VS Code muhiti uchun barqaror skript:
- EDA (namunalar, o'rtacha o'lcham, sinf taqsimoti, RGB gist.)
- Train/Test split (80/20)
- YOLO dataset.yaml yaratish
- (ixtiyoriy) YOLO o'qitish

Ishga tushirish:
    python main_yolo_pomidor.py
Yoki chizmalarni ko'rsatmasdan:
    python main_yolo_pomidor.py --no-show
"""

import os
import sys
import glob
import shutil
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch   # <<< qo‘shildi

# (ixtiyoriy) Ultralytics bor bo'lsa import, bo'lmasa o'tkazib yuboramiz
try:
    from ultralytics import YOLO  # noqa: F401
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False


# -----------------------------------------------------------
# Yordamchi funksiyalar
# -----------------------------------------------------------

SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(images_dir: str) -> List[str]:
    files = []
    for ext in SUPPORTED_IMG_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    files = sorted(files)
    return files


def label_path_for(image_path: str, labels_dir: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, base + ".txt")


def load_pairs(images_dir: str, labels_dir: str) -> Tuple[List[str], List[str]]:
    images = []
    labels = []
    for img_path in list_images(images_dir):
        lbl_path = label_path_for(img_path, labels_dir)
        if os.path.exists(lbl_path):
            images.append(img_path)
            labels.append(lbl_path)
    return images, labels


def read_yolo_label_lines(label_file: str) -> List[str]:
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        return lines
    except Exception:
        return []


def parse_class_ids_from_lines(lines: List[str]) -> List[int]:
    class_ids = []
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))
            class_ids.append(cid)
        except Exception:
            continue
    return class_ids


def calc_average_size(images: List[str]) -> Tuple[Optional[float], Optional[float]]:
    total_w, total_h, n = 0, 0, 0
    for p in images:
        try:
            with Image.open(p) as im:
                w, h = im.size
            total_w += w
            total_h += h
            n += 1
        except Exception:
            pass
    if n == 0:
        return None, None
    return total_w / n, total_h / n


def visualize_samples(images: List[str], num_samples: int = 5, show: bool = True):
    if not images:
        print("Vizualizatsiya uchun rasm topilmadi.")
        return
    k = min(num_samples, len(images))
    picks = random.sample(images, k)
    fig, axs = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axs = [axs]
    for ax, p in zip(axs, picks):
        img = cv2.imread(p)
        if img is None:
            ax.set_axis_off()
            ax.set_title("O'qib bo'lmadi")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(os.path.basename(p), fontsize=9)
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()


def plot_average_size(avg_w: float, avg_h: float, show: bool = True):
    cats = ["Average Width", "Average Height"]
    vals = [avg_w, avg_h]
    plt.figure(figsize=(6, 4))
    plt.bar(cats, vals)
    plt.ylabel("Pixels")
    plt.title("Average Image Dimensions")
    plt.ylim(0, max(vals) * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    if show:
        plt.show()
    plt.close()


def plot_class_distribution(label_paths: List[str], show: bool = True) -> Dict[str, int]:
    counts = {"unripe": 0, "ripe": 0}
    for lp in label_paths:
        lines = read_yolo_label_lines(lp)
        class_ids = parse_class_ids_from_lines(lines)
        for cid in class_ids:
            if cid == 0:
                counts["unripe"] += 1
            elif cid == 1:
                counts["ripe"] += 1
    print(f"ripe: {counts['ripe']}")
    print(f"unripe: {counts['unripe']}")
    plt.figure(figsize=(5, 4))
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.title("Class Distribution (instances)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    if show:
        plt.show()
    plt.close()
    return counts


def _read_image_cv(p: str) -> np.ndarray:
    img = cv2.imread(p)
    return img if img is not None else np.array([])


def process_images_in_batches(
    image_paths: List[str], max_images: int = 50, batch_size: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Rs, Gs, Bs = [], [], []
    subset = image_paths[: max_images]
    for i in range(0, len(subset), batch_size):
        batch = subset[i : i + batch_size]
        with ThreadPoolExecutor() as ex:
            imgs = list(ex.map(_read_image_cv, batch))
        for img in imgs:
            if img.size == 0:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            r, g, b = rgb[:, :, 0].ravel(), rgb[:, :, 1].ravel(), rgb[:, :, 2].ravel()
            Rs.append(r)
            Gs.append(g)
            Bs.append(b)
    if Rs:
        return np.concatenate(Rs), np.concatenate(Gs), np.concatenate(Bs)
    return np.array([]), np.array([]), np.array([])


def plot_rgb_histogram(
    image_paths: List[str],
    title_prefix: str,
    max_images: int = 50,
    batch_size: int = 10,
    show: bool = True,
):
    R, G, B = process_images_in_batches(image_paths, max_images, batch_size)
    if R.size == 0:
        print(f"{title_prefix}: gistogramma uchun rasm topilmadi.")
        return
    plt.figure(figsize=(14, 4))
    for idx, (ch, name) in enumerate(zip([R, G, B], ["Red", "Green", "Blue"])): 
        plt.subplot(1, 3, idx + 1)
        hist, bins = np.histogram(ch, bins=256, range=(0, 256))
        plt.bar(bins[:-1], hist, width=1, alpha=0.6)
        plt.title(f"{title_prefix} — {name}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()


def split_train_test(
    image_paths: List[str],
    label_paths: List[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    from sklearn.model_selection import train_test_split
    return train_test_split(
        image_paths, label_paths, test_size=test_size, random_state=seed, shuffle=True
    )


def safe_copy(files: List[str], dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for src in files:
        try:
            shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))
        except Exception as e:
            print(f"Copy xatosi: {src} -> {dst_dir}: {e}")


def write_dataset_yaml(
    yaml_path: str,
    train_images_dir: str,
    val_images_dir: str,
    class_names: List[str],
):
    nc = len(class_names)
    content = (
        f"train: {os.path.abspath(train_images_dir)}\n"
        f"val: {os.path.abspath(val_images_dir)}\n\n"
        f"nc: {nc}\n"
        f"names: {class_names}\n"
    )
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


def classify_images_by_presence(
    image_paths: List[str], label_paths: List[str]
) -> Tuple[List[str], List[str]]:
    ripe_images, unripe_images = [], []
    for img, lbl in zip(image_paths, label_paths):
        lines = read_yolo_label_lines(lbl)
        cids = set(parse_class_ids_from_lines(lines))
        if 1 in cids:
            ripe_images.append(img)
        if 0 in cids:
            unripe_images.append(img)
    return ripe_images, unripe_images


# -----------------------------------------------------------
# Asosiy oqim
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="rasmlar", help="Asosiy ma'lumotlar katalogi")
    parser.add_argument("--show", dest="show", action="store_true", help="Chizmalarni ko'rsatish")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Chizmalarni ko'rsatmaslik")
    parser.set_defaults(show=True)
    parser.add_argument("--max-hist-imgs", type=int, default=50, help="RGB gistogramma uchun eng ko'p rasm")
    args = parser.parse_args()

    base_data_dir = args.base
    images_dir = os.path.join(base_data_dir, "Images")
    labels_dir = os.path.join(base_data_dir, "labels")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("Xato: 'Images' yoki 'labels' papkasi topilmadi. Yo'lni tekshiring.")
        sys.exit(1)

    print(f"Images Dir: {images_dir}, Labels Dir: {labels_dir}")
    print("Ma'lumotlar papkalari topildi. Dastur davom ettirilmoqda.")

    # ---------------- EDA
    images, labels = load_pairs(images_dir, labels_dir)
    print(f"Topilgan rasmlar soni: {len(images)}")
    print(f"Topilgan yorliqlar soni: {len(labels)}")

    files = glob.glob(os.path.join(images_dir, "*"))
    exts = sorted({os.path.splitext(f)[1].lower() for f in files})
    print("Unique File Extensions:")
    for e in exts:
        print(e)

    visualize_samples(images, num_samples=5, show=args.show)

    avg_w, avg_h = calc_average_size(images)
    if avg_w is not None and avg_h is not None:
        print(f"Average Image Width: {avg_w:.2f} pixels")
        print(f"Average Image Height: {avg_h:.2f} pixels")
        plot_average_size(avg_w, avg_h, show=args.show)

    class_counts = plot_class_distribution(labels, show=args.show)

    ripe_imgs, unripe_imgs = classify_images_by_presence(images, labels)
    print(f"Number of ripe images: {len(ripe_imgs)}")
    print(f"Number of unripe images: {len(unripe_imgs)}")

    plot_rgb_histogram(ripe_imgs, "Ripe Tomatoes", max_images=args.max_hist_imgs, show=args.show)
    plot_rgb_histogram(unripe_imgs, "Unripe Tomatoes", max_images=args.max_hist_imgs, show=args.show)

    # ---------------- Train/Test split
    working_dir = "yolo_data"
    train_images_dir = os.path.join(working_dir, "train", "images")
    train_labels_dir = os.path.join(working_dir, "train", "labels")
    test_images_dir = os.path.join(working_dir, "test", "images")
    test_labels_dir = os.path.join(working_dir, "test", "labels")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    imgs_train, imgs_test, lbls_train, lbls_test = split_train_test(images, labels, test_size=0.2, seed=42)
    print(f"O'rgatish to'plami: {len(imgs_train)} rasm, {len(lbls_train)} yorliq")
    print(f"Test to'plami: {len(imgs_test)} rasm, {len(lbls_test)} yorliq")

    safe_copy(imgs_train, train_images_dir)
    safe_copy(lbls_train, train_labels_dir)
    safe_copy(imgs_test, test_images_dir)
    safe_copy(lbls_test, test_labels_dir)

    print("Ma'lumotlar muvaffaqiyatli ajratildi va papkalarga joylashtirildi.")

    # ---------------- dataset.yaml
    dataset_yaml_path = os.path.join(working_dir, "dataset.yaml")
    class_names = ["unripe", "ripe"]
    write_dataset_yaml(dataset_yaml_path, train_images_dir, test_images_dir, class_names)
    print("dataset.yaml fayli yaratildi.")

    # ---------------- YOLO o‘qitish
    if ULTRALYTICS_AVAILABLE:
        model = YOLO("yolov8n.pt")
        results = model.train(
            data="yolo_data/dataset.yaml",
            epochs=100,       # 130 emas, avval kichik son bilan test
            imgsz=320,       # kichik rasm o‘lchami → tezroq
            batch=2,         # kichik batch → RAM tejaydi
            device="cpu",    # GPU yo‘q, shuni ishlatish kerak
            workers=0        # dataloader workerlarini 0 qilamiz
        )
        model.export(format="onnx")   # masalan ONNX formatida
        model.export(format="torchscript") 
        print("O‘qitish tugadi ✅")

        val_results = model.val()
        print("Baholash tugadi ✅")

        sample_test_imgs = glob.glob(os.path.join(test_images_dir, "*.jpg")) + \
                           glob.glob(os.path.join(test_images_dir, "*.jpeg")) + \
                           glob.glob(os.path.join(test_images_dir, "*.png"))
        for img_path in sample_test_imgs[:5]:
            r = model(img_path, conf=0.5, iou=0.6)[0]
            im_array = r.plot()
            save_path = os.path.join(working_dir, "pred_" + os.path.basename(img_path))
            cv2.imwrite(save_path, im_array)
            print(f"Natija saqlandi: {save_path}")
    else:
        print("⚠️ Ultralytics (YOLOv8) o‘rnatilmagan. `pip install ultralytics` bilan o‘rnating.")

    print("\nDastur yakunlandi.")


if __name__ == "__main__":
    main()
