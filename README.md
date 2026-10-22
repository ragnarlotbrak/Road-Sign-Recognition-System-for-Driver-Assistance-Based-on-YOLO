# Road Sign Recognition System for Driver Assistance Based on YOLO

A system that detects and classifies road signs in images and video from a windshield camera and alerts the driver to relevant signs. To choose the base model, the project compares the lightweight YOLO models YOLOv8n, YOLOv10n and YOLO11n under identical settings.

Bachelor's thesis, Astana IT University, 2027. Group SE-2423.

## Problem Statement

Drivers can miss or misread road signs because of fatigue, distraction, unfamiliar roads, poor weather or low light. A missed sign, such as a speed limit, can lead to a traffic violation or an accident. Many vehicles have no built-in sign recognition. Existing solutions are either tied to proprietary hardware or rely on map data, which can be outdated and does not include temporary signs.

Camera-based detection is also difficult. Signs are often small and far away, detection accuracy drops at night, in rain and in fog, and the model must still run in real time on limited hardware. Published studies mostly propose modified versions of a single baseline model, so it remains unclear which standard lightweight YOLO version a developer should start from. This project builds an open and reproducible recognition module and selects its base model through a controlled comparison.

## Target Users and Expected Outcomes

The target users are drivers of vehicles without factory sign recognition who use a smartphone or a dashcam, and developers who need an open sign recognition module for their own driver assistance applications. Evidence from the target users is being collected through a survey and interviews (see Documentation).

The expected outcomes are:

1. A trained road sign detection model, selected after comparing YOLOv8n, YOLOv10n and YOLO11n under identical training settings.
2. An evaluation report with precision, recall and mAP@0.5 for clear, night, rain and fog conditions, together with inference speed (FPS and latency) measured against a real-time threshold of 30 FPS.
3. A prototype API that processes images and video and returns detected signs and alerts.
4. A documented repository with tests and run instructions.

## Team and Supervisor

| Name | Role | Responsibilities |
|---|---|---|
| Kassenov Abdulkarim | Software Engineer | Video pipeline, API and application |
| Ryskali Mansur | Data Engineer | Dataset collection, cleaning and validation |
| Atalykov Sultan | ML Engineer | Model training, comparison and evaluation |

Supervisor: Kamila Tanyrbergenova

## Scope

In scope: detection and classification of 20 road sign classes, including speed limits, stop, give way, no entry, no overtaking, danger ahead, sharp curves and turn right (the full list is in the dataset plan). The system accepts an image, a video file or a camera stream as input, produces visual alerts, and is evaluated separately under clear, night, rain and fog conditions.

Out of scope: vehicle control, lane detection, other driver assistance functions, and optimisation for embedded in-vehicle boards. Speed is measured on the prototype hardware.

## Current Status

The project is at the analysis and planning stage.

Completed: topic selection, a literature review of 12 sources, the research question, aim, objectives and hypotheses, dataset assembly and validation, and a YOLOv8n baseline trained in Google Colab. An initial FastAPI service is in the `app` folder.

In progress: a user survey (Google Forms, in Kazakh, Russian and English), interviews with three drivers with different levels of experience, the comparison of existing systems and technologies, requirements and acceptance criteria, architecture diagrams, and the verification plan.

Next steps: building separate test subsets for clear, night, rain and fog conditions, training YOLOv10n and YOLO11n under the same settings as the baseline, connecting the selected model to the API, and adding automated tests. Baseline metrics are printed by the notebook and will be added to this README together with the results of YOLOv10n and YOLO11n.

## Dataset

The training data contains 4,770 images in 20 road sign classes in YOLO format, built from two public Kaggle datasets. Images are not stored in Git. The dataset can be downloaded from [Google Drive link]. Access is restricted and is given to the team, the supervisor and the committee on request.

Two archives are available. `road_signs_yolo.zip` is the clean dataset. `road_signs_yolo_aug.zip` is the same dataset with augmented training images. Its validation and test sets are identical to the clean version, so both versions are evaluated on the same test data.

The dataset is imbalanced, with a ratio of about 20 to 1 between the most and the least frequent classes, so the evaluation reports per-class metrics. Details, the class table and limitations are in [docs/data/dataset-plan.md](docs/data/dataset-plan.md).

## Selected Technologies

Training and evaluation use Python 3, Ultralytics (YOLOv8n, YOLOv10n, YOLO11n) and PyTorch in Google Colab with a T4 GPU. The application uses OpenCV for reading images and video, and FastAPI with Uvicorn for the API. Tests are written with pytest. Version control is Git and GitHub, and tasks are tracked in YouTrack. The justification of each choice is in the technology comparison.

`requirements.txt` lists the dependencies of the API. The notebook installs the training dependencies itself.

## Documentation

| Document | Location |
|---|---|
| Project proposal and scope | [docs/proposal.md](docs/proposal.md) |
| Literature review, analysis matrix and search log | [docs/literature-review/](docs/literature-review/) |
| Research question, aim, objectives and hypotheses | [docs/research-question.md](docs/research-question.md) |
| Evidence of relevance (survey and interviews) | [docs/evidence/](docs/evidence/) |
| Existing system comparison | [docs/existing-systems.md](docs/existing-systems.md) |
| Technology comparison | [docs/technology-comparison.md](docs/technology-comparison.md) |
| Requirements and acceptance criteria | [docs/requirements.md](docs/requirements.md) |
| Verification and evaluation plan | [docs/verification-plan.md](docs/verification-plan.md) |
| Dataset plan | [docs/data/dataset-plan.md](docs/data/dataset-plan.md) |
| Diagrams (editable sources and exports) | [diagrams/](diagrams/) |

## Repository Structure

```text
app/                                    FastAPI service
docs/                                   Proposal, literature review, comparisons, requirements, plans
  data/                                 Dataset plan and class table
diagrams/                               Editable diagram sources and exported images
models/                                 Trained model weights (e.g. best.pt), not tracked by Git
tests/                                  Automated tests
dataset_check_and_baseline_test.ipynb   Dataset check and YOLOv8n baseline (Google Colab)
requirements.txt                        API dependencies
```

The `uploads` and `results` folders are created by the API at runtime and are not tracked by Git.

## How to Run

### Dataset check and baseline in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ragnarlotbrak/Road-Sign-Recognition-System-for-Driver-Assistance-Based-on-YOLO/blob/main/dataset_check_and_baseline_test.ipynb)

1. Open the notebook in Google Colab using the badge above.
2. Choose Runtime → Change runtime type → T4 GPU.
3. Set the `DATASET_URL` variable to the dataset link (access is required, see Dataset).
4. Choose Runtime → Run all.

The notebook downloads the dataset, validates the labels, plots the class distribution, trains a YOLOv8n baseline, and prints precision, recall, mAP and inference speed.

### API on a local machine

1. Clone the repository and open its folder:

```bash
   git clone https://github.com/ragnarlotbrak/Road-Sign-Recognition-System-for-Driver-Assistance-Based-on-YOLO.git
   cd Road-Sign-Recognition-System-for-Driver-Assistance-Based-on-YOLO
```

2. Download the trained model weights (`best.pt`) from [Google Drive link] and place the file in the `models/` folder.

3. Create and activate a virtual environment:

```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS / Linux
```

4. Install the dependencies and start the server:

```bash
   pip install -r requirements.txt
   uvicorn app.main:app --reload
```

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try the endpoints.

## Testing

The notebook performs the dataset checks: every image has a matching label file, class ids are in range, box coordinates are normalised, and no image appears in more than one of the train, validation and test sets. Automated pytest tests will be added to the `tests` folder and run with `pytest` from the repository root.

## Workflow

Each task has a YouTrack ID. Work is done in a branch named after this ID, for example `DIP-8-dataset-validation-notebook`, and commit messages start with the ID. Changes are merged into `main` through a pull request reviewed by another team member. Each pre-defense submission is marked with a Git tag (`pre-defense-1`, `pre-defense-2`, `final-pre-defense`). Secrets such as Kaggle keys are never committed.

## Links

YouTrack project: [rsr-diploma.youtrack.cloud]

Dataset: [Google Drive link]