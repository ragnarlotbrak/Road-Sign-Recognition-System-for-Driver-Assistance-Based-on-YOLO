#!/usr/bin/env python3
"""
build_dataset.py - build one clean YOLO road-sign dataset from two source datasets.

What it does:
  1. Reads both datasets (YOLO format, 416x416) and unifies the class names.
  2. Finds duplicates with a perceptual hash (dHash 16x16):
       - true duplicates (the same picture)  -> one copy is kept, boxes are merged;
       - near-identical frames (simulator sequences, frames of one GTSRB sign) -> kept,
         but always placed in the same split (otherwise train -> val/test data leakage).
  3. Drops rare classes (too few independent image groups).
  4. Drops images where one sign carries two different classes (label noise).
  5. Splits into train/valid/test by GROUPS, stratified by class.
  6. Writes the dataset, data.yaml, manifest.csv and REPORT.md, then checks for leakage.

Usage:
  python build_dataset.py --ds1 path/to/car --ds2 path/to/TrafficSignLocalizationandDetection \
                          --ds2-yaml path/to/Custom_data.yaml --out out/road_signs_yolo
Requires: numpy, pandas, opencv-python, pyyaml
"""
import argparse, collections, hashlib, os, random, re, shutil
import cv2, numpy as np, pandas as pd, yaml

SPLITS = ["train", "valid", "test"]
RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}
DUP_HAMMING = 3          # d <= 3 of 256 bits: the same picture (inside one dataset)
NEAR_HAMMING = 8         # d <= 8: similar frames -> one split group
MIN_GROUPS = 30          # a class is kept if it has >= 30 independent image groups
INCLUDE_TRAFFIC_LIGHTS = False   # traffic lights are not signs; set True to keep them
MIN_BOX_PX = 2           # boxes smaller than 2 px (at 416) are treated as junk
SEED = 42
LIGHTS = {"Green Light", "Red Light"}


def norm(name):
    return re.sub(r"\s+", " ", name.replace("-", "")).strip()


def dhash(path, n=16):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    s = cv2.resize(im, (n + 1, n), interpolation=cv2.INTER_AREA)
    return (s[:, 1:] > s[:, :-1]).flatten()


def hamming_pairs(H, max_d):
    P = np.packbits(H, axis=1)
    pc = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    out = []
    for i in range(len(P) - 1):
        d = pc[np.bitwise_xor(P[i + 1:], P[i])].sum(1)
        for k in np.where(d <= max_d)[0]:
            out.append((i, i + 1 + int(k), int(d[k])))
    return out


class UF:
    def __init__(self, n): self.p = list(range(n))
    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def u(self, a, b): self.p[self.f(a)] = self.f(b)


def read_labels(path):
    """-> list of (class_id, xc, yc, w, h); polygons are converted to a bbox."""
    out, bad = [], 0
    if not os.path.exists(path):
        return out, bad
    for line in open(path).read().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0])); v = list(map(float, p[1:]))
        if len(v) > 4:  # polygon
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            v = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        x, y, w, h = v[:4]
        x1, y1, x2, y2 = max(0, x - w / 2), max(0, y - h / 2), min(1, x + w / 2), min(1, y + h / 2)
        w, h = x2 - x1, y2 - y1
        if w * 416 < MIN_BOX_PX or h * 416 < MIN_BOX_PX:
            bad += 1; continue
        out.append((c, (x1 + x2) / 2, (y1 + y2) / 2, w, h))
    return out, bad


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def add_boxes(dst, src_boxes, thr):
    for b in src_boxes:
        if not any(d[0] == b[0] and iou(d[1:], b[1:]) > thr for d in dst):
            dst.append(b)


def load(ds1, ds2, ds2_yaml):
    n1 = yaml.safe_load(open(os.path.join(ds1, "data.yaml")))["names"]
    n2 = yaml.safe_load(open(ds2_yaml))["names"]
    rows, dropped_boxes = [], 0
    for src, root, names in [("ds1", ds1, n1), ("ds2", ds2, n2)]:
        for sp in SPLITS:
            idir = os.path.join(root, sp, "images")
            for f in sorted(os.listdir(idir)):
                stem = f.rsplit(".", 1)[0]
                lab, bad = read_labels(os.path.join(root, sp, "labels", stem + ".txt"))
                dropped_boxes += bad
                rows.append(dict(src=src, orig_split=sp, file=f, img=os.path.join(idir, f),
                                 base=f.split(".rf.")[0],
                                 boxes=[(norm(names[b[0]]),) + b[1:] for b in lab]))
    return pd.DataFrame(rows), dropped_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds1", required=True); ap.add_argument("--ds2", required=True)
    ap.add_argument("--ds2-yaml", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(SEED)

    df, tiny = load(a.ds1, a.ds2, a.ds2_yaml)
    N = len(df); n_raw = N
    print(f"loaded {N} images, tiny boxes dropped: {tiny}")

    # --- 1. hashes and duplicates ---
    H = np.stack([dhash(p) for p in df.img])
    md5 = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in df.img]
    pairs = hamming_pairs(H, NEAR_HAMMING)
    dup, grp = UF(N), UF(N)
    src = df.src.values
    n_cross_ds = 0
    for i, j, d in pairs:
        grp.u(i, j)
        if d <= DUP_HAMMING or src[i] != src[j]:
            dup.u(i, j)
            n_cross_ds += src[i] != src[j]
    for idx in pd.Series(range(N)).groupby(md5).apply(list):
        for k in idx[1:]:
            dup.u(idx[0], k); grp.u(idx[0], k)
    # frames of one physical GTSRB sign (class_track_frame) -> one group
    tr = df.base.str.extract(r"^(\d{5})_(\d{5})_\d{5}_png")
    df["track"] = np.where(tr[0].notna(), tr[0] + "_" + tr[1], None)
    for _, idx in df[df.track.notna()].groupby("track").groups.items():
        idx = list(idx)
        for k in idx[1:]: grp.u(idx[0], k)
    df["dup"] = [dup.f(i) for i in range(N)]
    df["grp"] = [grp.f(i) for i in range(N)]

    # --- 2. which classes to keep (counted by independent groups) ---
    cls_groups = collections.defaultdict(set)
    for r in df.itertuples():
        for b in r.boxes:
            cls_groups[b[0]].add(r.grp)
    keep = {c for c, g in cls_groups.items() if len(g) >= MIN_GROUPS}
    if not INCLUDE_TRAFFIC_LIGHTS:
        keep -= LIGHTS
    classes = sorted(keep)
    cid = {c: i for i, c in enumerate(classes)}
    removed_classes = sorted(set(cls_groups) - keep)

    # --- 3. one image per duplicate cluster, boxes are merged ---
    items = []   # final images
    n_empty = 0; n_conflict = 0
    for d, g in df.groupby("dup"):
        members = list(g.itertuples())
        fl = lambda r: [b for b in r.boxes if b[0] in keep]
        members.sort(key=lambda r: (-len(fl(r)), r.src, r.file))
        rep = members[0]
        boxes = []
        add_boxes(boxes, fl(rep), 0.9)
        for m in members[1:]:                     # boxes from a copy in the other dataset
            if m.src != rep.src:
                add_boxes(boxes, fl(m), 0.5)
        if not boxes:
            n_empty += 1; continue
        # one sign labelled with two different classes -> label noise, drop the image
        if any(boxes[i][0] != boxes[j][0] and iou(boxes[i][1:], boxes[j][1:]) > 0.5
               for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
            n_conflict += 1; continue
        items.append(dict(rep=rep, boxes=boxes, grp=rep.grp, n_merged=len(members),
                          srcs="+".join(sorted({m.src for m in members}))))

    # --- 4. group-based stratified split ---
    groups = collections.defaultdict(list)
    for it in items: groups[it["grp"]].append(it)
    gcnt = {}
    for gk, its in groups.items():
        c = collections.Counter(cid[b[0]] for it in its for b in it["boxes"])
        gcnt[gk] = c
    total = collections.Counter()
    for c in gcnt.values(): total.update(c)
    target = {s: {k: RATIOS[s] * v for k, v in total.items()} for s in SPLITS}
    cur = {s: collections.Counter() for s in SPLITS}
    n_img = {s: 0 for s in SPLITS}; tgt_img = {s: RATIOS[s] * len(items) for s in SPLITS}
    order = sorted(groups, key=lambda gk: (min(total[k] for k in gcnt[gk]), random.random()))
    assign = {}
    for gk in order:
        best, bs = None, -1e9
        for s in SPLITS:
            sc = sum(v * (target[s][k] - cur[s][k]) / max(target[s][k], 1) for k, v in gcnt[gk].items())
            sc += 0.5 * (tgt_img[s] - n_img[s]) / max(tgt_img[s], 1)
            if sc > bs: best, bs = s, sc
        assign[gk] = best
        cur[best].update(gcnt[gk]); n_img[best] += len(groups[gk])

    # --- 5. write output ---
    out = a.out
    if os.path.exists(out): shutil.rmtree(out)
    for s in SPLITS:
        os.makedirs(f"{out}/images/{s}"); os.makedirs(f"{out}/labels/{s}")
    man = []
    for it in items:
        r = it["rep"]; s = assign[it["grp"]]
        name = f"{r.src}_{r.file}"; stem = name.rsplit(".", 1)[0]
        shutil.copy2(r.img, f"{out}/images/{s}/{name}")
        with open(f"{out}/labels/{s}/{stem}.txt", "w") as f:
            for b in it["boxes"]:
                f.write(f"{cid[b[0]]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
        man.append(dict(file=name, split=s, source=r.src, original_split=r.orig_split,
                        group=it["grp"], merged_copies=it["n_merged"], sources_merged=it["srcs"],
                        n_boxes=len(it["boxes"])))
    pd.DataFrame(man).to_csv(f"{out}/manifest.csv", index=False)
    yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/valid",
                    "test": "images/test", "nc": len(classes), "names": classes},
                   open(f"{out}/data.yaml", "w"), sort_keys=False, allow_unicode=True)

    # --- 6. leakage check on the OUTPUT files ---
    mdf = pd.DataFrame(man)
    oh = np.stack([dhash(f"{out}/images/{r.split}/{r.file}") for r in mdf.itertuples()])
    leak = [(i, j) for i, j, d in hamming_pairs(oh, NEAR_HAMMING) if mdf.split[i] != mdf.split[j]]
    trk = {f"{r.src}_{r.file}": r.track for r in df.itertuples() if r.track}
    mdf["track"] = mdf.file.map(trk)
    tspan = (mdf[mdf.track.notna()].groupby("track").split.nunique() > 1).sum()
    ndup = len(mdf) - mdf.file.nunique()
    print(f"check: cross-split leaks = {len(leak)}, GTSRB tracks in >1 split = {tspan}")
    assert not leak and tspan == 0 and ndup == 0, "data leakage!"

    # --- 7. report ---
    rows = []
    for c in classes:
        k = cid[c]
        per = {s: cur[s][k] for s in SPLITS}
        ng = len({it["grp"] for it in items if any(b[0] == c for b in it["boxes"])})
        note = "few samples" if per["valid"] < 10 or per["test"] < 10 else ""
        rows.append(f"| {k} | {c} | {total[k]} | {per['train']} | {per['valid']} | {per['test']} | {ng} | {note} |")
    cnt = mdf.groupby("split").size()
    rep = f"""# Dataset build report

Sources (both from Kaggle): ds1 = "Traffic Signs Detection" by pkdarabi (a copy of Roboflow "Self-Driving Cars" v6,
CC BY 4.0, 4969 images, 15 classes), ds2 = "Traffic-Sign-Localization-Detection-YOLO-Annotated" by ahemateja19bec1025
(1738 images, 36 classes, license not stated). Both YOLO format, 416x416 (stretch).

## What was done
- Input images: **{n_raw}**
- Duplicates (the same picture, incl. ds2 copies inside ds1) merged: **-{n_raw - len({d for d in df.dup}) }**; ds1<->ds2 matching pairs: {n_cross_ds}
- Images left without boxes after class filtering: {n_empty}
- Images dropped because one sign carries two different classes (label noise): {n_conflict}
- Tiny boxes (< {MIN_BOX_PX} px) dropped: {tiny}
- Near-identical frames (simulator sequences, frames of one GTSRB sign) are kept but always stay in one split
- Split: group-based, stratified by class, {RATIOS['train']:.0%}/{RATIOS['valid']:.0%}/{RATIOS['test']:.0%}, seed {SEED}
- Traffic lights (Green/Red Light) {'included' if INCLUDE_TRAFFIC_LIGHTS else 'excluded (they are not signs); switch with INCLUDE_TRAFFIC_LIGHTS'}
- Classes removed (< {MIN_GROUPS} independent image groups): {', '.join(removed_classes)}

## Result
- **Images: {len(mdf)}** (train {cnt['train']}, valid {cnt['valid']}, test {cnt['test']})
- **Classes: {len(classes)}**
- Leakage check on the output files: 0 similar pairs across splits, 0 GTSRB tracks in several splits

| id | Class | Boxes | train | valid | test | Groups | Note |
|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + """

## Notes for training
- **fliplr=0.0** (and no rotations): a mirror turns Turn Right into Turn Left, swaps Left/Right Sharp Curve, and 80 into 08.
- "Groups" = number of independent sources. Speed Limit 20 has many images but few groups (frames of the same sign).
- Classes marked "few samples" cannot be evaluated reliably on valid/test.
"""
    open(f"{out}/REPORT.md", "w").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
