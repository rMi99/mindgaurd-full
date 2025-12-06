# University Research Project Mentor Guide: MindGuard Enhanced

This guide is designed to help you elevate your BEng (Hons) software engineering project to meet and exceed university research standards. It analyzes your current codebase, identifies gaps, and provides a concrete roadmap for improvement.

---

## 1. Key Criteria for a Successful Research Project

University projects (especially "Top-up" or final year) are graded differently from standard industry projects. They require:

1.  **Critical Analysis & Problem Solving**: Not just *building* a feature, but explaining *why* you built it that way and comparing it to alternatives.
2.  **Robust Methodology**: A clear, scientific approach to designing, building, and **evaluating** your system.
3.  **Quantitative Evaluation**: You must prove your system works using data (graphs, metrics, comparisons), not just "it runs without errors".
4.  **Academic Depth**: Use of design patterns, algorithms, or novel architectures (which you have started, but need to complete).
5.  **Reflection**: Discussing limitations and future work honestly.

---

## 2. Gap Analysis: Where is your project now?

I have analyzed your codebase (`backend/app`, `scripts`, `Dockerfile`). Here is the honest assessment:

### ✅ Strengths
-   **Ambitious Architecture**: You have outlined a sophisticated system using **Observer**, **Factory**, and **Strategy** patterns.
-   **Modern Stack**: Next.js 14, FastAPI, Docker, and Python 3.12 is a very strong, industry-standard stack.
-   **"Adaptive" Concept**: The idea of switching models (CNN vs MobileNet) based on accuracy/overfitting is excellent research material.

### ⚠️ Critical Gaps (The "danger zone" for grading)
1.  **Missing Model Implementations**:
    -   Your `FacialModelFactory` (`backend/app/core/patterns.py`) tries to import `CNNModel`, `MobileNetModel`, etc., from `app.models`, but **these files do not exist**.
    -   You have `enhanced_model.py`, but it is a *Health Risk* model (tabular data), not a *Facial Analysis* model.
    -   **Impact**: Your "Dynamic Model Switching" feature—a core research claim—currently cannot work because the models aren't there.

2.  **The "Adaptive" Loop is Broken**:
    -   Your `DropoutTuningStrategy` detects overfitting and returns an action (e.g., "increase_dropout"), but the system just logs it. It doesn't actually *change* the model's parameters in real-time.
    -   **Impact**: The system is "Monitoring" but not truly "Adaptive".

3.  **Lack of Quantitative Evaluation**:
    -   You have a training script (which I improved), but you lack a **System Evaluation** script. How do you prove that "Adaptive" mode is better than "Static" mode?
    -   **Requirement**: You need a comparison: `System A (Static)` vs `System B (Adaptive)`.

4.  **Testing**:
    -   The test suite was largely empty. A research project needs robust unit and integration tests to prove reliability.

---

## 3. Step-by-Step Improvement Plan

To turn this into a First-Class (70%+) project, follow this roadmap:

### Phase 1: Fix the Core "Research" Feature (Implementation)
*Goal: Make the "Dynamic Model Switching" actually work.*

1.  **Create the Missing Model Classes**:
    -   Create `backend/app/models/base_facial_model.py` (Abstract Base Class).
    -   Create `backend/app/models/cnn_model.py` (A heavy, accurate model, e.g., ResNet or VGG).
    -   Create `backend/app/models/mobilenet_model.py` (A light, fast model).
    -   *Hint*: Use the `torchvision` models (like I used in `train_facial_model.py`) but wrap them in your `AIModel` class structure.

2.  **Connect the Factory**:
    -   Update `backend/app/core/patterns.py` to correctly import these new classes.

3.  **Implement Real Switching**:
    -   Ensure `AdaptiveFacialAnalysisService` actually calls `self.model_manager.switch_model()` when a threshold is triggered.

### Phase 2: rigorous Evaluation (The "Research" Part)
*Goal: Generate the data for your dissertation.*

1.  **Create an Evaluation Script**:
    -   Write a script `scripts/evaluate_adaptive_system.py`.
    -   It should run a dataset of ~100 images through the system **twice**:
        -   **Run 1 (Baseline)**: Fixed model (e.g., MobileNet only). Measure Accuracy and Latency.
        -   **Run 2 (Adaptive)**: Your system. Measure Accuracy and Latency.
    -   **Hypothesis**: Adaptive mode should match the accuracy of the heavy model but have lower average latency (or better accuracy than the light model).

2.  **Generate Graphs**:
    -   "Accuracy vs. Time"
    -   "Latency vs. Model Type"
    -   Save these for your final report.

### Phase 3: Documentation & Professionalism
1.  **Architecture Diagram**:
    -   Draw a UML Class Diagram showing how `AdaptiveService` uses `ModelFactory` and `AccuracyMonitor`.
    -   Draw a Sequence Diagram showing the "Overfitting Detected -> Switch Model" flow.

2.  **API Documentation**:
    -   Ensure FastAPI's `/docs` are clean and described.

---

## 4. Concrete Advice for Maximizing Marks

*   **Innovation**: The "Overfitting Detection" on live data is your "Secret Sauce". Emphasize this. Most student projects just train a model once. You are monitoring it *live*.
*   **Complexity**: Explicitly mention your use of Design Patterns (Strategy, Observer, Factory) in your report. This shows "Software Engineering" maturity.
*   **Validation**: If you can't get real user data (ethics issues), use a "Replay Mechanism" where you feed a video file into the system and treat it as a live camera. This is a perfectly valid research validation method.

---

## 5. Next Actions for You

1.  **Immediate**: Implement `backend/app/models/mobilenet_model.py` and `cnn_model.py` so your factory stops crashing.
2.  **Short Term**: Run the `train_facial_model.py` script I fixed to generate actual `.pt` model files to load.
3.  **Long Term**: Write the `evaluate_adaptive_system.py` script to generate your graphs.

Good luck! This project has the potential to be excellent if you close the gap between your *architecture* (patterns) and your *implementation* (actual models).
