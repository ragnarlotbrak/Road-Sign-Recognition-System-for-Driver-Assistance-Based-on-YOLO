# Dataset Plan: Data Collection, Annotation and Augmentation

**Project:** Road Sign Recognition System for Driver Assistance Based on YOLO
**Role:** Data Engineer (data collection, annotation, augmentation)
**Author:** _<Ryskali Mansur>_ | **YouTrack task:** _<RSR-1>_ | **Date:** 22 Oct 2026

---

## 1. Goal and scope

Prepare a dataset for training a YOLO detector that finds road signs in an image and recognises their type.

| Parameter | Value |
|---|---|
| Annotation format | YOLO: `class_id x_center y_center width height`, values in 0..1 |
| Image size | 416×416 (stretch resize, as in the sources) |
| Final size (before augmentation) | **4,770 images**, 5,282 boxes |
| Classes | **20** |
| Split | train 3,326 / valid 717 / test 727 (≈ 70 / 15 / 15) |
| Train after augmentation | 6,652 (originals + 1 copy of each); valid and test unchanged |

## 2. Data sources

| | Dataset 1 (ds1) | Dataset 2 (ds2) |
|---|---|---|
| Name | Kaggle "Traffic Signs Detection" (author: pkdarabi); a copy of Roboflow "Self-Driving Cars" v6 | Kaggle "Traffic-Sign-Localization-Detection-YOLO-Annotated" (author: ahemateja19bec1025) |
| Images | 4,969 | 1,738 (the dataset page mentions ~2,500) |
| Classes | 15 (speed limits, stop, traffic lights) | 36 (warning, prohibitory, mandatory, speed limits) |
| License | CC BY 4.0 (stated in the dataset's own `data.yaml`, inherited from Roboflow) | **not stated in the downloaded files**; the author collected the images from various sources |
| Link | https://www.kaggle.com/datasets/pkdarabi/cardetection (original: https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou) | https://www.kaggle.com/datasets/ahemateja19bec1025/trafficsignlocalizationdetectionyoloannotated |

**Reviewed and rejected:** a very large "Traffic Sign Detection" dataset (Arnad_77). It was found to be unsuitable for this task.

## 3. Data processing (reproducible pipeline)

Script: `src/data/build_dataset.py` (fixed `seed=42`).

1. **Unified class list.** Class names were normalised and class ids renumbered. Without this, the same sign had different ids in the two datasets (for example, Stop is 14 in ds1 and 31 in ds2).
2. **Duplicate search with a perceptual hash (dHash 16×16).** Comparing raw files was not enough: copies re-encoded as JPEG have different bytes but are the same picture. 704 of the 1,738 ds2 images already exist in ds1.
   - true duplicates were merged into one image, boxes from the copies were combined;
   - 6,707 → 5,636 images (−1,071 duplicates; 794 ds2 ↔ ds1 pairs among the matches found).
3. **Data leakage prevention.** Near-identical frames (simulator sequences, frames of one physical GTSRB sign) are kept but always placed in the **same** split.
4. **Class filtering.** Classes with ≥ 30 independent image groups were kept. Traffic lights were excluded (they are not road signs). Of the 42 unique classes in the two sources, 22 were removed (including the two traffic lights), among them Pedestrian, Road Work, RoundAbout, Turn Left and Speed Limit 10.
5. **Box filtering.** 125 boxes smaller than 2 px were dropped.
6. **Conflicting labels removed.** If boxes of different classes overlap on one sign (IoU > 0.5), the whole image is dropped. There were 5 such images (for example, a sign "30" labelled both "30" and "50"); the correct class is visible by eye but cannot be chosen automatically.
7. **Split** into train / valid / test by groups, stratified by class.
8. **Verification of the result.** On the output files: 0 similar pairs across splits, 0 GTSRB sign series in several splits, all labels within valid bounds, every image has a label file, 0 overlapping boxes with different classes.

> The first merge attempt contained 522 pairs of near-identical images in different splits, which would have inflated the metrics. It also kept the ds1–ds2 duplicates and spread 206 of 314 GTSRB sign series over several splits. These problems were found during re-checking and fixed in the current version.

## 4. Classes

| id | Class | Boxes | train | valid | test | Note |
|---|---|---|---|---|---|---|
| 0 | Danger Ahead | 40 | 28 | 6 | 6 | few samples |
| 1 | Give Way | 73 | 51 | 11 | 11 | |
| 2 | Left Sharp Curve | 45 | 31 | 7 | 7 | few samples |
| 3 | No Entry | 81 | 57 | 12 | 12 | |
| 4 | No Over Taking | 65 | 43 | 11 | 11 | |
| 5 | No Over Taking Trucks | 48 | 33 | 8 | 7 | few samples |
| 6 | Right Sharp Curve | 36 | 24 | 7 | 5 | few samples |
| 7 | Speed Limit 100 | 399 | 276 | 59 | 64 | |
| 8 | Speed Limit 110 | 120 | 84 | 18 | 18 | |
| 9 | Speed Limit 120 | 389 | 270 | 61 | 58 | |
| 10 | Speed Limit 20 | 256 | 185 | 35 | 36 | few independent groups (35) |
| 11 | Speed Limit 30 | 529 | 367 | 79 | 83 | |
| 12 | Speed Limit 40 | 341 | 238 | 51 | 52 | |
| 13 | Speed Limit 50 | 461 | 323 | 69 | 69 | |
| 14 | Speed Limit 60 | 415 | 286 | 67 | 62 | |
| 15 | Speed Limit 70 | 484 | 339 | 73 | 72 | |
| 16 | Speed Limit 80 | 466 | 326 | 69 | 71 | |
| 17 | Speed Limit 90 | 216 | 150 | 33 | 33 | |
| 18 | Stop | 702 | 485 | 108 | 109 | |
| 19 | Turn Right | 116 | 81 | 18 | 17 | |

## 5. Augmentation

Script: `src/data/augment_dataset.py` (`seed=42`, 1 copy per image by default).

**What is applied (colour and quality changes only; the position of the sign does not change):** brightness and contrast, hue and saturation shift, Gaussian blur, motion blur, noise, fog, rain, low light (dusk), JPEG compression. Each image gets 1–2 randomly chosen distortions.

**What is NOT applied, and why:**
- flips and rotations: a mirror turns "Turn Right" into "Turn Left" and "Left Sharp Curve" into "Right Sharp Curve", and a rotation turns "80" into "08";
- augmentation of valid and test: otherwise the metrics would not be honest.

The author of ds2 gives the same warning on the dataset page: rotational augmentation can turn 80 into 08 or a right turn into a left turn and hurt the model.

Because the geometry does not change, the label files of the copies are identical to the originals. Copies stay in train, so there is no data leakage. The `augmentation_log.csv` file shows which distortion was applied to each copy.

Note: the low-light distortion is fairly strong. 112 of the 3,326 copies (3.4%) end up very dark (mean brightness below 20 of 255); 88 of them include the low-light distortion, the rest started from dark originals. The sign is still visible in them.

## 6. Recommendations for training (for the ML engineer)

- Pass `fliplr=0.0` and `degrees=0` to the training run. These are training arguments (Ultralytics uses `fliplr=0.5` by default); they cannot be set in `data.yaml`.
- **Baseline:** YOLO nano fine-tuned on this dataset **without offline augmentation** (original train images only, `road_signs_yolo`). A COCO-pretrained model without fine-tuning is not suitable for comparison: of our 20 classes it only knows Stop.
- **Experiment:** the same model on train with augmentation (`road_signs_yolo_aug`). Model, number of epochs, seed and image size are identical; only the data changes. Ultralytics already applies online augmentation (mosaic, colour shifts and others), so the gain from offline copies may be small, and that is an acceptable result.
- **Optional:** the COCO model without fine-tuning, evaluated on the Stop class only, to show the benefit of fine-tuning.
- Metrics: precision, recall, mAP@0.5, mAP@0.5:0.95, and **per-class** metrics.
- Use the test split once, at the end. Tune parameters on valid.
- Treat classes marked "few samples" with caution: with 5–8 examples in valid and test the metrics are noisy.
- For the verification plan: evaluate small signs separately (about a third of all boxes, 32%, cover less than 1% of the image area).

## 7. Known limitations

1. **Class imbalance.** 11 of the 20 classes are speed limits, and four other classes have only 36–48 examples.
2. **Not all sign categories are covered.** There is no pedestrian crossing, "Road Work", "Roundabout" and others: there was not enough data for them (see 3.4). The system recognises 20 classes, not all signs.
3. **Unlabelled signs.** Classes that do not exist in ds1 (for example Give Way, No Entry) are labelled only on ds2 images. On ds1 images such signs may be present without a box, and the model may learn to treat them as background.
4. **Heterogeneous data.** The dataset mixes GTSRB crops (the sign fills almost the whole image), ordinary photos, and fisheye simulator frames where signs are very small.
5. **Geography.** Signs are mostly European. Their match with Kazakhstani signs has not been checked.
6. **Resolution.** Images were stretched to 416×416, so fine details of distant signs are lost.
7. **Residual label noise.** Only conflicts of the type "one sign, two classes" were detected automatically (and removed). Single wrong labels (for example a red warning triangle among "Turn Right" in ds2) were not searched for.
8. **License and rights.** ds2 has no stated license, and some images carry stock-photo watermarks. The dataset is for educational use only and must not be published or redistributed. ds1 (CC BY 4.0) requires attribution.

## 8. Repository structure and data storage

```
src/data/
  build_dataset.py       # dataset build and cleaning
  augment_dataset.py     # train augmentation
docs/data/
  dataset-plan.md        # this document
data.yaml                # class list
```

Images are **not stored in Git**. The dataset is kept on _<Google Drive: https://drive.google.com/drive/folders/1HLAWxdLGdXSkmfgzKq2AbTzR2fnwHCEC?usp=sharing>_ with restricted access (team and supervisor only), because ds2 has no license. The dataset archive contains `data.yaml`, `manifest.csv` (origin of every image: source dataset, original split, merged copies) and `REPORT.md`.

Reproduction:

```bash
python src/data/build_dataset.py --ds1 <path>/car \
    --ds2 <path>/TrafficSignLocalizationandDetection \
    --ds2-yaml <path>/Custom_data.yaml --out data/road_signs_yolo
python src/data/augment_dataset.py --src data/road_signs_yolo \
    --out data/road_signs_yolo_aug --copies 1
```

## 9. Possible next steps

- Collect a small own test set (50–100 photos of real signs in Kazakhstan) and check how well the model generalises to it.
- Bring back the rare classes if an additional data source appears.
- Label the unlabelled signs on ds1 images (see 7.3).
- Make augmentation class-aware: more copies for rare classes (Danger Ahead, Left/Right Sharp Curve, No Over Taking Trucks).
