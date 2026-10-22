#!/usr/bin/env python3
"""
augment_dataset.py - safe augmentation of the TRAIN part of the YOLO road-sign dataset.

Principles:
  * ONLY train is augmented. valid and test are copied as they are (otherwise metrics are not honest).
  * Only PHOTOMETRIC distortions are used (brightness, contrast, blur, noise, fog, rain,
    low light, JPEG compression). Geometry is not changed, so the label (.txt) files are
    copied unchanged.
  * NO flips and NO rotations: a mirror turns Turn Right into Turn Left, and 80 into 08.
  * Copies stay in the same split (train) as the original, so there is no data leakage.

Usage:
  python augment_dataset.py --src road_signs_yolo --out road_signs_yolo_aug --copies 1
Requires: numpy, opencv-python, pyyaml
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png"}


# ---------- individual distortions (input and output: uint8 BGR, same size) ----------

def brightness_contrast(img, rng):
    alpha = rng.uniform(0.7, 1.3)    # contrast
    beta = rng.uniform(-40, 40)      # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def gaussian_blur(img, rng):
    k = rng.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img, rng):
    k = rng.choice([3, 5, 7])
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k                         # horizontal streak
    angle = rng.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    kernel = kernel / s if s > 0 else kernel
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img, rng, np_rng):
    sigma = rng.uniform(5, 20)
    noise = np_rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def fog(img, rng):
    h, w = img.shape[:2]
    strength = rng.uniform(0.2, 0.5)
    # fog is denser near the horizon (top of the frame) and weaker at the bottom
    grad = np.linspace(1.0, 0.4, h, dtype=np.float32)[:, None, None]
    haze = np.full_like(img, 215, dtype=np.uint8).astype(np.float32)
    out = img.astype(np.float32) * (1 - strength * grad) + haze * strength * grad
    return np.clip(out, 0, 255).astype(np.uint8)


def rain(img, rng, np_rng):
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.uint8)
    n = int(rng.uniform(0.0005, 0.0015) * h * w)
    slant = rng.randint(-4, 4)
    for _ in range(n):
        x, y = np_rng.integers(0, w), np_rng.integers(0, h)
        length = rng.randint(4, 10)
        cv2.line(layer, (x, y), (x + slant, y + length), 255, 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    streaks = cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(img, 1.0, streaks, rng.uniform(0.25, 0.5), 0)
    return cv2.GaussianBlur(out, (3, 3), 0)  # rain slightly blurs the frame


def low_light(img, rng):
    gamma = rng.uniform(1.6, 2.4)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
    out = cv2.LUT(img, lut)
    return cv2.convertScaleAbs(out, alpha=rng.uniform(0.6, 0.9), beta=0)


def jpeg_compress(img, rng):
    q = rng.randint(25, 60)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def color_shift(img, rng):
    """Small hue and saturation shift (different lighting and cameras)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_random(img, rng, np_rng):
    """Picks 1-2 distortions. Returns the image and the list of their names."""
    ops = [
        ("brightness_contrast", lambda i: brightness_contrast(i, rng)),
        ("color_shift", lambda i: color_shift(i, rng)),
        ("gaussian_blur", lambda i: gaussian_blur(i, rng)),
        ("motion_blur", lambda i: motion_blur(i, rng)),
        ("noise", lambda i: gaussian_noise(i, rng, np_rng)),
        ("fog", lambda i: fog(i, rng)),
        ("rain", lambda i: rain(i, rng, np_rng)),
        ("low_light", lambda i: low_light(i, rng)),
        ("jpeg", lambda i: jpeg_compress(i, rng)),
    ]
    chosen = rng.sample(ops, rng.choice([1, 2]))
    for _, fn in chosen:
        img = fn(img)
    return img, [name for name, _ in chosen]


# ---------- main script ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset folder (images/, labels/, data.yaml)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--copies", type=int, default=1, help="augmented copies per train image")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    if out.exists():
        shutil.rmtree(out)
    # 1. valid and test: copied unchanged
    for sp in ["valid", "test"]:
        shutil.copytree(src / "images" / sp, out / "images" / sp)
        shutil.copytree(src / "labels" / sp, out / "labels" / sp)
    # 2. train: originals + augmented copies
    (out / "images" / "train").mkdir(parents=True)
    (out / "labels" / "train").mkdir(parents=True)

    log = ["file,source_file,ops"]
    n_orig = n_aug = 0
    for img_path in sorted((src / "images" / "train").iterdir()):
        if img_path.suffix.lower() not in IMG_EXT:
            continue
        lbl = src / "labels" / "train" / (img_path.stem + ".txt")
        shutil.copy2(img_path, out / "images" / "train" / img_path.name)
        shutil.copy2(lbl, out / "labels" / "train" / lbl.name)
        n_orig += 1

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        for k in range(args.copies):
            aug, ops = apply_random(img, rng, np_rng)
            name = f"{img_path.stem}__aug{k + 1}.jpg"
            cv2.imwrite(str(out / "images" / "train" / name), aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
            shutil.copy2(lbl, out / "labels" / "train" / f"{img_path.stem}__aug{k + 1}.txt")
            log.append(f"{name},{img_path.name},{'+'.join(ops)}")
            n_aug += 1

    # 3. data.yaml and the augmentation log
    cfg = yaml.safe_load(open(src / "data.yaml", encoding="utf-8"))
    cfg["path"] = "."
    yaml.safe_dump(cfg, open(out / "data.yaml", "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    (out / "augmentation_log.csv").write_text("\n".join(log) + "\n", encoding="utf-8")

    n_val = len(list((out / "images" / "valid").iterdir()))
    n_test = len(list((out / "images" / "test").iterdir()))
    print(f"train: {n_orig} originals + {n_aug} augmented = {n_orig + n_aug}")
    print(f"valid: {n_val} (unchanged), test: {n_test} (unchanged)")
    print("Remember when training: fliplr=0.0, degrees=0 (no flips, no rotations).")


if __name__ == "__main__":
    main()
