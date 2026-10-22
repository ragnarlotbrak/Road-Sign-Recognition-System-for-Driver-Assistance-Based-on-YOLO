# Dataset build report

Sources (both from Kaggle): ds1 = "Traffic Signs Detection" by pkdarabi (a copy of Roboflow "Self-Driving Cars" v6,
CC BY 4.0, 4969 images, 15 classes), ds2 = "Traffic-Sign-Localization-Detection-YOLO-Annotated" by ahemateja19bec1025
(1738 images, 36 classes, license not stated). Both YOLO format, 416x416 (stretch).

## What was done
- Input images: **6707**
- Duplicates (the same picture, incl. ds2 copies inside ds1) merged: **-1071**; ds1<->ds2 matching pairs: 794
- Images left without boxes after class filtering: 861
- Images dropped because one sign carries two different classes (label noise): 5
- Tiny boxes (< 2 px) dropped: 125
- Near-identical frames (simulator sequences, frames of one GTSRB sign) are kept but always stay in one split
- Split: group-based, stratified by class, 70%/15%/15%, seed 42
- Traffic lights (Green/Red Light) excluded (they are not signs); switch with INCLUDE_TRAFFIC_LIGHTS
- Classes removed (< 30 independent image groups): Cycle Zone, Deer Zone, End of Right Road Go straight, Go Left or Straight, Go Right or Straight, Go Straight, Green Light, Huddle Road, Left Curve Ahead, No Stopping, No Waiting, Pedestrian, Red Light, Right Curve Ahead, Road Work, RoundAbout, Slippery Road, Snow Warning Sign, Speed Limit 10, Traffic Signals Ahead, Truck Sign, Turn Left

## Result
- **Images: 4770** (train 3326, valid 717, test 727)
- **Classes: 20**
- Leakage check on the output files: 0 similar pairs across splits, 0 GTSRB tracks in several splits

| id | Class | Boxes | train | valid | test | Groups | Note |
|---|---|---|---|---|---|---|---|
| 0 | Danger Ahead | 40 | 28 | 6 | 6 | 35 | few samples |
| 1 | Give Way | 73 | 51 | 11 | 11 | 60 |  |
| 2 | Left Sharp Curve | 45 | 31 | 7 | 7 | 42 | few samples |
| 3 | No Entry | 81 | 57 | 12 | 12 | 61 |  |
| 4 | No Over Taking | 65 | 43 | 11 | 11 | 41 |  |
| 5 | No Over Taking Trucks | 48 | 33 | 8 | 7 | 43 | few samples |
| 6 | Right Sharp Curve | 36 | 24 | 7 | 5 | 32 | few samples |
| 7 | Speed Limit 100 | 399 | 276 | 59 | 64 | 286 |  |
| 8 | Speed Limit 110 | 120 | 84 | 18 | 18 | 89 |  |
| 9 | Speed Limit 120 | 389 | 270 | 61 | 58 | 216 |  |
| 10 | Speed Limit 20 | 256 | 185 | 35 | 36 | 35 |  |
| 11 | Speed Limit 30 | 529 | 367 | 79 | 83 | 353 |  |
| 12 | Speed Limit 40 | 341 | 238 | 51 | 52 | 324 |  |
| 13 | Speed Limit 50 | 461 | 323 | 69 | 69 | 285 |  |
| 14 | Speed Limit 60 | 415 | 286 | 67 | 62 | 212 |  |
| 15 | Speed Limit 70 | 484 | 339 | 73 | 72 | 233 |  |
| 16 | Speed Limit 80 | 466 | 326 | 69 | 71 | 349 |  |
| 17 | Speed Limit 90 | 216 | 150 | 33 | 33 | 181 |  |
| 18 | Stop | 702 | 485 | 108 | 109 | 399 |  |
| 19 | Turn Right | 116 | 81 | 18 | 17 | 111 |  |

## Notes for training
- **fliplr=0.0** (and no rotations): a mirror turns Turn Right into Turn Left, swaps Left/Right Sharp Curve, and 80 into 08.
- "Groups" = number of independent sources. Speed Limit 20 has many images but few groups (frames of the same sign).
- Classes marked "few samples" cannot be evaluated reliably on valid/test.
