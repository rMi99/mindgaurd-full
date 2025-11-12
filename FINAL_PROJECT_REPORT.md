# MindGaurd — Final Project Report

> Note: This document is a formalized and extended final-project / thesis-style report for the MindGaurd repository (multimodal mental-health assessment). It combines research/literature-review sections with a practical implementation and deployment guide referencing the repository's code and scripts.

## Title

MindGaurd: Multimodal AI for Mental Health Assessment from Facial, Voice and Survey Data

Authors: rMi99 (project repository: `mindgaurd-full`)

Supervisor / Course: [Add supervisor and course]

Date: [Add submission date]

## Abstract

This project develops and evaluates MindGaurd, a multimodal system to assist mental health assessment by combining facial-expression analysis, voice analysis, and survey (PHQ-9-style) signals. The system integrates pre-processing pipelines, deep learning models for vision and audio, a lightweight backend API for inference, and a Next.js frontend for user interaction. The research component investigates prior work in affective computing, multimodal fusion strategies, and ethical considerations for deploying AI in healthcare. The implementation demonstrates training and inference scripts, experimental results, deployment via Docker Compose, and reproducible instructions for evaluation.

## Keywords

multimodal, affective computing, facial analysis, voice analysis, mental health, PHQ-9, inference API, deployment, reproducibility

## Table of contents

- Abstract
- Introduction
- Research and Literature Review
- Datasets
- System architecture and components
- AI models and algorithms
- Implementation details (code references)
- Experimental setup and results
- Deployment and APIs
- Ethics, privacy and limitations
- Conclusions and future work
- References
- Appendices (run instructions, API spec, hyperparameters, code snippets)

## Introduction

Motivation: mental health conditions such as depression are underdiagnosed. Automated multimodal assessment tools can help clinicians and researchers by providing objective quantified signals while preserving clinical oversight. MindGaurd aims to provide a reproducible research prototype combining facial, audio, and questionnaire inputs to estimate mental-health risk indicators.

Project goals:

- Build and train models for facial expression and voice features related to mood and depression.
- Fuse multimodal signals to produce a risk score or classification aligned with PHQ-9 categories.
- Provide reproducible training and inference code, APIs, and deployment artifacts.
- Evaluate performance, discuss limitations, and address ethical/privacy concerns.

Assumptions

- This report assumes access to the repository `mindgaurd-full` and its scripts (see `scripts/`, `backend/`, `frontend/`).
- The project focuses on proof-of-concept and research, not clinical deployment.

## Research and Literature Review

This section summarizes the related research fields and positions the project within them.

1. Affective computing and facial expression analysis

- Ekman's foundational work on facial action units (FAUs) and emotion recognition. Modern deep learning approaches (CNNs, transfer learning with models like EfficientNet, ResNet, and deepface variants) are state-of-the-art for facial emotion recognition.
- Relevant recent works: DeepFace, OpenFace, FER datasets (FER2013), AffectNet, and papers on depression detection from facial dynamics.

2. Speech and vocal biomarkers

- Acoustic features (MFCCs, pitch, energy, spectral features) correlate with mood and depression. Modern approaches use CNNs or LSTMs on spectrograms or raw waveforms, or transformer-based audio models.

3. Multimodal fusion

- Fusion strategies: early fusion (concatenate features), late fusion (ensemble outputs), and joint representation learning. Research suggests careful modality alignment and attention mechanisms can improve performance.

4. Clinical considerations and evaluation

- Use of PHQ-9 as validated screening tool. Sensitivity/specificity trade-offs and the need for clinical oversight.
- Ethics literature warns against unvalidated automated diagnosis, privacy risks of biometric data, and bias in ML models.

Where this project fits

MindGaurd implements modular pipelines for facial and audio feature extraction, trains deep models (or fine-tunes pretrained networks), and explores simple fusion strategies. The research part reviews the literature above, proposes hypotheses (e.g., multimodal fusion improves PHQ-9 category prediction), and tests them experimentally.

## Datasets

List of datasets used or recommended (from the repo):

- Local CSV datasets in `data/` such as `phq9.csv`, `mental_health_survey.csv`, `dataset.csv` (inspect and document columns before use).
- Public facial/emotion datasets for pretraining: FER2013, AffectNet, RAF-DB.
- Audio datasets: publicly available speech emotion corpora (e.g., CREMA-D, RAVDESS) and any internal voice recordings produced under consent.

Data handling and preprocessing

- Document consent and de-identification steps.
- For facial pipelines: detect and crop faces, align, resize (e.g., 224x224), normalize per pretrained model requirements.
- For audio pipelines: resample to 16 kHz, remove silence, compute MFCCs or log-mel spectrograms.
- For survey data: parse PHQ-9 totals and map to label classes (none/minimal/mild/moderate/severe).

## System architecture and components

High-level components (repo mapping):

- Frontend: Next.js app at `frontend/app/` that provides assessment UI and dashboards.
- Backend API: FastAPI/Flask app entry in `backend/app/main.py` and related routes in `backend/app/routes/`.
- Training and inference scripts: `scripts/train_model.py`, `scripts/infer_model.py`, and modality-specific scripts (e.g., `train_facial_model.py`, `train_audio_model.py`).
- Data and models: `data/` for datasets and `data/models/` for saved weights.
- Orchestration: Dockerfiles and `docker-compose.*.yml` for local development and production.

Architecture diagram (text)

User/browser -> Next.js frontend -> Backend API (/api/assess, /api/infer) -> Model runner (Python) -> Data store / logs

## AI models and algorithms

This project uses separate modality models and a fusion module.

1) Facial model

- Backbone: pretrained convolutional network (e.g., ResNet50 / EfficientNet / a deepface variant)
- Head: classification/regression head for mood/depression score
- Loss: cross-entropy (classification) or MSE (regression)
- Files: `scripts/train_facial_model.py`, `scripts/train_fer_model.py`

2) Audio model

- Input: MFCCs or log-mel spectrograms
- Model: CNN / CNN+LSTM or transformer-based audio encoder
- Files: `scripts/train_audio_model.py`, `scripts/train_model.py` (general)

3) Text/Survey model

- Input: PHQ-9 numeric score or item vectors
- Model: simple MLP or logistic regression to map to categories

4) Fusion strategy

- Late fusion baseline: take modality logits/probabilities and average or train a small combiner MLP on concatenated logits.
- Early fusion: concatenate learned embeddings from each modality and train a joint classifier.

Suggested evaluation metrics

- Accuracy, precision, recall, F1-score, ROC-AUC for classification.
- Mean Absolute Error (MAE) or RMSE for regression on continuous symptom scores.

## Software & design patterns used

This section documents the software engineering and architectural patterns used in MindGaurd and recommended patterns for AI projects. Including these in your thesis demonstrates the engineering rigor behind model design, deployment, and maintenance.

1) Architectural patterns

- Modular pipeline (separation of concerns): data ingestion, preprocessing, model training, inference, and serving are implemented as separate modules or services. Example locations: `scripts/` (training/inference), `backend/app/` (serving), `frontend/` (UI).
- Microservice or service-oriented style (optional): backend API can be split into model-serving, user-management, and analytics services for scalability. Docker Compose files in the repo illustrate service separation.

2) Common software design patterns in the codebase

- Factory pattern: used to create data loaders or model instances based on config (e.g., a single factory function that returns a facial/audio dataset loader depending on a `modality` parameter). Where to look: training scripts (`scripts/`) where dataset and model are constructed from config.
- Strategy pattern: for interchangeable preprocessing or augmentation strategies (e.g., different face-augmentation pipelines or audio preprocessing pipelines). Implement as classes or functions that conform to a preprocessing interface.
- Adapter pattern: to make third-party model APIs or pre-trained model loaders (DeepFace, torchaudio transforms) conform to the project's expected model or dataset interfaces.
- Repository pattern (data access): abstract data sources (CSV, DB, S3) behind a consistent interface to make experiments reproducible and storage backends swappable.
- Observer / Event pattern: for extensible logging or callbacks during training (e.g., a callback system that triggers model checkpointing, metric logging, or early stopping).

3) MLOps and model lifecycle patterns

- Experiment tracking and reproducibility: run tracking (Weights & Biases, MLflow, or simple CSV logs) to capture hyperparameters, metrics, and artifacts.
- Model registry pattern: central store (or directory) for model versions, metadata, and provenance.
- CI/CD for models: automated unit tests, linting, and integration tests for data processing and model serving; automated container image builds and deployment pipelines.
- Canary and staged rollout: deploy models to a subset of traffic or users, monitor metrics before full rollout.

4) Deployment patterns

- Containerization: Dockerfiles for each component and `docker-compose.*.yml` for orchestration.
- Sidecar monitoring: use a sidecar or separate service for metrics collection and log shipping (Prometheus + Grafana recommended).

5) Security & privacy-by-design

- Encryption at rest and in transit for sensitive media and survey data.
- Least-privilege access control for backend endpoints and model artifacts.
- Data minimization: keep only the features and artifacts needed for research and delete raw media where possible.

6) Testing and validation patterns

- Unit tests for data processing functions, integration tests for end-to-end inference, and smoke tests for API endpoints. Add tests under `backend/` and `scripts/tests/` where appropriate.

Concrete examples and repo pointers

- Factory/Strategy: check `scripts/train_model.py` and modality-specific training scripts for how model and dataset objects are constructed from config flags.
- Docker-based deployment: `docker-compose.dev.yml` and `Dockerfile.*` files.
- Inference endpoints: `backend/app/routes/` and `backend/app/main.py`.

## Research methodology and what to include in the research part

The research section should move beyond literature review to a clear, reproducible experimental plan. Include the following subsections and items in your thesis:

1) Research questions and hypotheses

- Clearly state primary and secondary research questions. Example: "Does multimodal fusion of facial, audio, and survey signals improve classification of PHQ-9 depression categories compared to single-modality baselines?"
- State hypotheses with expected directionality (e.g., multimodal fusion will yield higher F1 by X percentage points).

2) Study design

- Dataset selection and inclusion/exclusion criteria.
- Train/validation/test split protocol and subject-disjoint splitting when applicable.
- Data augmentation and preprocessing specifics.

3) Experimental protocol

- Baselines and ablation studies to run (face-only, audio-only, survey-only, late fusion, early fusion).
- Exact hyperparameter ranges and search strategy (grid/random/Bayesian), number of seeds, early stopping criteria.

4) Statistical analysis

- Significance testing (paired t-tests, bootstrap, or non-parametric tests) for performance differences between models.
- Confidence intervals for key metrics and effect sizes.

5) Evaluation metrics and error analysis

- Use both aggregate metrics (F1, AUC) and per-class metrics. Present confusion matrices.
- Error analysis: qualitative review of false positives/negatives and correlation with metadata (e.g., demographics, recording quality).

6) Reproducibility and documentation

- Exact commands used for training and evaluation, random seeds, and environment details (Python version, package versions in `backend/requirements.txt`).
- Link to model checkpoints and experiment logs (or include hashes if large).

7) Ethical approvals and consent

- Document IRB/ethics approvals if data was collected, informed consent forms, and data handling procedures.

8) Limitations and threats to validity

- External validity: dataset representativeness.
- Internal validity: leakage between splits, labeling noise.

## Implementation details (code references)

## Implementation details (code references)

Key files to inspect/modify:

- `backend/app/main.py` — backend server entry and API wiring.
- `backend/requirements.txt` — Python dependencies required for the backend.
- `scripts/train_model.py` and modality-specific training scripts — contain training loops and data loaders.
- `scripts/infer_model.py` — example inference script for running models on new inputs.
- `frontend/app/` — Next.js UI components for assessment and dashboards.
- `docker-compose.dev.yml` and `Dockerfile.*` — development and production deployment definitions.

How code is organized

- Modular: separate scripts per modality, a central backend to serve endpoints, and a Next.js frontend to collect inputs and display results.
- Models are expected to save and load weights from `data/models/`.

Example: training flow for facial model

1. Prepare dataset CSV (face paths + labels) in `data/`.
2. Run `python scripts/train_facial_model.py --config configs/facial.yaml` (adapt flags available in script).
3. Monitor training logs, save model to `data/models/facial_best.pth`.

## Experimental setup and results

Experimental design

- Split datasets into train/val/test (e.g., 70/15/15) ensuring subject disjointness where necessary.
- Baselines: single-modality models (face-only, audio-only, survey-only) and multimodal fusion.
- Run multiple seeds and report mean ± std for metrics.

Hyperparameters (example)

- Learning rate: 1e-4 (backbone fine-tune: 1e-5)
- Batch size: 16-64 (memory dependent)
- Optimizer: AdamW
- Epochs: 20-100 depending on dataset size

Results (how to present)

- Use confusion matrices and ROC curves for classification.
- Present table comparing modalities and fusion performance.
- Report computational costs (training time, inference latency) and model sizes.

Reproducibility

- Record random seed, environment (Python version, package versions in `backend/requirements.txt` and `frontend/package.json`), and model checkpoints.

## Deployment and APIs

Docker and orchestration

- The repository contains `docker-compose.dev.yml` and other Dockerfiles to run the backend and frontend.
- For local development, run the `docker-compose.dev.yml` stack (services: backend, frontend, optionally nginx). See `start.sh` and `start_assessment_test.sh` for helpers.

API design (example spec)

- POST /api/assess
  - Description: runs a full assessment using supplied media (image, audio) and questionnaire answers.
  - Request (multipart/form-data):
    - `image` (optional): face image or video frame
    - `audio` (optional): audio file (wav/mp3)
    - `survey` (optional): JSON with PHQ-9 answers
  - Response (application/json):
    - `facial_score`: float
    - `audio_score`: float
    - `survey_score`: float
    - `fusion_score`: float
    - `labels`: predicted class labels and confidences

- POST /api/infer
  - Single-model inference endpoint, accepts `modality` param and corresponding input.

Backend endpoints are implemented in `backend/app/routes/` — inspect this folder to confirm exact route names and payload shapes.

Authentication and privacy

- Protect endpoints with authentication for any production deployment. Consider token-based auth (JWT) and role-based access for clinicians vs researchers.

## Ethics, privacy and limitations

Ethical considerations

- Biomedical disclaimer: this system is not a diagnostic tool. It provides risk indicators that should be interpreted by qualified clinicians.
- Consent and data governance: record informed consent for any participant data. Anonymize or encrypt stored media and logs.

Bias and fairness

- Visual and acoustic models can pick up demographic biases. Evaluate per-group performance (age, gender, skin tone) and report disparities.

Limitations

- Dataset size and representativeness may limit generalizability.
- Models may be sensitive to recording conditions (lighting, microphone quality).

Mitigations

- Data augmentation, domain adaptation, and collecting diverse samples.

## Conclusions and future work

Summary

- MindGaurd demonstrates a modular, reproducible pipeline for multimodal mental-health risk assessment combining facial, audio, and questionnaire signals. The repository includes training scripts, inference code, Docker-based deployment, and a Next.js front end.

Future directions

- Improve fusion with attention-based multimodal transformers.
- Integrate longitudinal tracking and calibration with clinical outcomes.
- Run user studies under IRB approval to collect labeled data and validate clinical utility.

## References

Add proper citations here. Suggested references to include (replace with full bibliographic entries as needed):

- Ekman, P. (Facial action coding system) — on facial expressions.
- Goodfellow et al., relevant deep learning for images.
- Papers on vocal biomarkers for depression (e.g., Cummins et al.).
- PHQ-9 validation papers.

Include the provided report `3. Software Engineering -Final Project Report Template .docx.pdf` in your bibliography if it contains sources used during writing.

## Appendices

Appendix A — How to run locally (dev)

1. Install Docker and Docker Compose.
2. From repo root, start dev stack:

```bash
# run from repository root
docker compose -f docker-compose.dev.yml up --build
```

3. Alternatively you can start the backend and frontend individually (see `start.sh`).

Appendix B — Quick API examples

Example request for inference (curl, multipart form):

```bash
curl -X POST "http://localhost:8000/api/assess" \
  -F "image=@/path/to/face.jpg" \
  -F "audio=@/path/to/audio.wav" \
  -F "survey={\"q1\":1,\"q2\":0,\"q3\":2}" \
  -H "Content-Type: multipart/form-data"
```

Example JSON response:

```json
{
  "facial_score": 0.32,
  "audio_score": 0.41,
  "survey_score": 0.27,
  "fusion_score": 0.35,
  "labels": {"depression_risk": "moderate", "confidence": 0.78}
}
```

Appendix C — Hyperparameter example table

| Component | Learning rate | Batch size | Optimizer | Notes |
|---|---:|---:|---|---|
| Facial model | 1e-4 | 32 | AdamW | backbone pretrained |
| Audio model | 1e-3 | 64 | Adam | spectrogram input |
| Fusion combiner | 1e-3 | 128 | Adam | small MLP on logits |

Appendix D — Important repository files and where to look

- `backend/app/main.py` — backend server
- `backend/requirements.txt` — Python dependencies
- `scripts/train_facial_model.py`, `scripts/train_audio_model.py`, `scripts/infer_model.py` — training and inference code
- `frontend/` — Next.js UI and components
- `docker-compose.*.yml` and `Dockerfile.*` — deployment

Appendix E — Recommended next steps for thesis write-up

1. Replace placeholders (supervisor, dates) and add your name.
2. Populate References with full citations and any literature discovered during research.
3. Copy experimental numbers, plots, and tables from your actual runs into the Results section.
4. Consider adding a dedicated Methods subsection with exact architectures, layer sizes, and a reproducible training recipe (seed, exact command line for training).

---

If you'd like, I can:

- Extract and cite specific references from your provided PDF and insert them in the References section.
- Generate a short Methods appendix with exact code snippets from `scripts/`.
- Create a LaTeX-ready version or exportable PDF.

Please tell me which of the above you'd like next.
