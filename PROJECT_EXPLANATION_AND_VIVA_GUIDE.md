# MindCare AI : Complete Project Guide & Viva Defense Handbook
**Project Title:** Predicting Depression and Mental Health Issues  
**Academic Course:** Minor Project MO/26 (Semester 7)  
**Department:** Department of Computer Science & Engineering, BIT Mesra Off-Campus Deoghar  
**Group No:** 22  
**Group Members:** Piyush Raj, Ipsita Gayatri, Mohit Kumar, Rajesh Kumar Mahato  

---

## Table of Contents
1. [Project Overview in Simple Terms (Layman Explanation)](#1-project-overview-in-simple-terms)
2. [Dataset Architecture & Key Properties](#2-dataset-architecture--key-properties)
3. [The Class Imbalance Challenge & Engineering Solution](#3-the-class-imbalance-challenge--engineering-solution)
4. [Data Preprocessing Pipeline](#4-data-preprocessing-pipeline)
5. [Feature Selection Optimization: A* Search vs. PSO](#5-feature-selection-optimization-a-search-vs-pso)
6. [Machine Learning & Deep Learning Model Architectures](#6-machine-learning--deep-learning-model-architectures)
7. [Explainable AI (XAI) via SHAP](#7-explainable-ai-xai-via-shap)
8. [Master Experimental Results Benchmark](#8-master-experimental-results-benchmark)
9. [Direct Answers to Synopsis Research Questions (RQ1, RQ2, RQ3)](#9-direct-answers-to-synopsis-research-questions)
10. [MindCare AI Web Application Architecture](#10-mindcare-ai-web-application-architecture)
11. [Comprehensive Viva & Oral Defense Questions (with Model Answers)](#11-comprehensive-viva--oral-defense-questions)
12. [Group Member Presentation Strategy (Who Speaks What)](#12-group-member-presentation-strategy)

---

## 1. Project Overview in Simple Terms

### What is this project?
Depression is traditionally diagnosed through subjective psychiatric clinical interviews. However, people often experience workplace stress, sleep disturbances, financial pressure, and lifestyle decline months before formal clinical diagnosis.

**Our goal:** Build an intelligent decision-support system that takes structured lifestyle, demographic, and occupational data and predicts whether an individual exhibits depression risk.

### What did we build?
1. **End-to-End Data Science Pipeline:** Evaluates 2,054 records across 11 personal and occupational variables.
2. **Two Optimization Search Algorithms:** A* Heuristic Search and Particle Swarm Optimization (PSO) to find the smallest, most predictive subset of features.
3. **Five Classifiers:** Evaluated Support Vector Machines (SVM), Random Forest, XGBoost, AdaBoost, and a 1D Convolutional Neural Network (CNN).
4. **Explainable AI (SHAP):** Unpacks the "black box" to explain *why* a model flagged a patient as high risk.
5. **Interactive Web Application:** A deployed, clinical screening tool called **MindCare AI** where doctors or employees can input attributes and receive real-time probabilistic risk stratification.

---

## 2. Dataset Architecture & Key Properties

- **File Name:** [`Depression Professional Dataset.csv`](file:///c:/Users/BIT/Desktop/boa/Depression%20Professional%20Dataset.csv)
- **Total Records:** 2,054
- **Total Features:** 11 (10 independent variables + 1 target variable)
- **Missing Values:** 0
- **Duplicate Records:** 0

### Variable Breakdown:
| Feature Name | Type | Categories / Scale | Meaning |
| :--- | :--- | :--- | :--- |
| `Gender` | Categorical | Female, Male | Demographic factor |
| `Age` | Numerical | 18 to 65 years | Demographic factor |
| `Work Pressure` | Numerical / Ordinal | 1.0 to 5.0 (Low to Severe) | Occupational strain |
| `Job Satisfaction` | Numerical / Ordinal | 1.0 to 5.0 (Low to High) | Workplace psychological health |
| `Sleep Duration` | Categorical | Less than 5h, 5-6h, 7-8h, More than 8h | Behavioral & biological factor |
| `Dietary Habits` | Categorical | Healthy, Moderate, Unhealthy | Nutritional factor |
| `Have you ever had suicidal thoughts ?` | Categorical | Yes, No | Critical psychiatric alert factor |
| `Work Hours` | Numerical | 0 to 14 hours/day | Occupational fatigue factor |
| `Financial Stress` | Numerical / Ordinal | 1 to 5 (Minimal to Critical) | Socioeconomic stressor |
| `Family History of Mental Illness` | Categorical | Yes, No | Genetic / familial vulnerability |
| `Depression` (Target) | Categorical Binary | **No (1,851 / 90.12%)**, **Yes (203 / 9.88%)** | Ground-truth diagnostic label |

---

## 3. The Class Imbalance Challenge & Engineering Solution

### The Core Problem:
Out of 2,054 people, only **203 (9.88%)** are depressed, while **1,851 (90.12%)** are healthy.

> **Crucial Viva Concept:**  
> If a "dumb" dummy model predicts "NO" for every single person, it achieves **90.12% accuracy**, but its clinical utility is **0%** because it misses every single depressed patient (0% Recall).

### How We Solved It:
1. **Stratified Splitting:** Used `stratify=y` during train/test split. Both the 80% training set and 20% testing set maintain the exact 90.12% vs. 9.88% ratio.
2. **Minority Penalty Balancing (`scale_pos_weight`):**
   $$\text{scale\_pos\_weight} = \frac{\text{Negative Samples}}{\text{Positive Samples}} = \frac{1,481}{162} \approx 9.14$$
   Every time the model misclassifies a depressed patient, it receives a **9.14x higher loss penalty** than misclassifying a non-depressed patient.
3. **Class Weights:** In SVM and Random Forest, we passed `class_weight='balanced'`. In 1D-CNN, we passed a class-weight matrix to `model.fit()`.
4. **Primary Metric Focus:** We evaluated models using **Recall** (Sensitivity to catch depressed patients), **Precision**, **F1-Score** (harmonic mean), and **ROC-AUC** rather than relying solely on raw Accuracy.

---

## 4. Data Preprocessing Pipeline

To eliminate data leakage, all transformations were fitted **strictly on the training split** (`X_train`) and applied downstream to the test split (`X_test`):

1. **Numerical Feature Transformer:**
   - Applied `StandardScaler()` on `Age`, `Work Pressure`, `Job Satisfaction`, `Work Hours`, `Financial Stress`.
   - Scaled values to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$).
2. **Categorical Feature Transformer:**
   - Applied `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` on `Gender`, `Sleep Duration`, `Dietary Habits`, `Suicidal Thoughts`, `Family History`.
   - Expanded the categorical variables into binary indicator columns.
3. **Output Feature Space:**
   - Combined through `ColumnTransformer` to yield **18 processed numerical features**.

---

## 5. Feature Selection Optimization: A* Search vs. PSO

Why do feature selection? Not all 18 features are equally predictive. Some add noise and increase computation. We compared two advanced search paradigms:

### Approach A: A* Heuristic Search (Deterministic State Space Search)
- **Concept:** An informed graph search algorithm using $f(n) = g(n) + h(n)$.
- **State Representation:** A node is a unique subset of feature indices, e.g., `(Age, Work Pressure, Suicidal Thoughts)`.
- **Cost Function $g(n)$:** Error rate on internal validation set plus a small penalty for each additional feature:
  $$g(n) = (1.0 - \text{F1}(n)) + 0.005 \times |n|$$
- **Heuristic $h(n)$:** Optimistic estimate of potential performance gain from the best remaining unused features.
- **Priority Queue:** Implemented using `heapq` in Python. Expanded states with the lowest $f(n)$ first.
- **Outcome:** Evaluated 120 subsets in 12.83 seconds, selecting **10 optimal features** with **Validation F1 = 91.18%**.

### Approach B: Particle Swarm Optimization (PSO - Swarm Intelligence)
- **Concept:** A metaheuristic inspired by the social foraging behavior of bird flocks.
- **Representation:** Discrete Binary PSO (`BinaryPSO`). A particle's position is an 18-element binary vector $[1, 0, 1, 1, ...]$ where $1$ denotes a selected feature.
- **Velocity & Sigmoid Transfer:** Velocity is updated based on personal best ($p_{\text{best}}$) and global best ($g_{\text{best}}$), converted to selection probabilities via a sigmoid function:
  $$S(v_{id}) = \frac{1}{1 + e^{-v_{id}}}$$
- **Objective Function:**
  $$\text{Cost} = (1.0 - \text{Validation F1}) + 0.005 \times \left(\frac{\text{selected features}}{18}\right)$$
- **Outcome:** 15 particles over 20 iterations completed in 9.20 seconds, selecting **10 features** with **Validation F1 = 88.89%**.

---

## 6. Machine Learning & Deep Learning Model Architectures

We evaluated 5 diverse model families:

1. **Support Vector Machine (SVM):**
   - Kernel: Radial Basis Function (RBF).
   - Maps non-linear tabular interactions into infinite-dimensional Hilbert space to find the maximum-margin hyperplane. Balanced class weights ensure the decision boundary does not collapse into the majority class.
2. **Random Forest Classifier:**
   - Ensemble of 200 de-correlated decision trees with bootstrap aggregation (bagging) and randomized feature subsets at each split.
3. **XGBoost (Extreme Gradient Boosting):**
   - Sequential boosting algorithm optimizing a second-order Taylor expansion of binary log-loss. Built-in L1/L2 regularization prevents overfitting.
4. **AdaBoost Classifier:**
   - Adaptive boosting that iteratively increases weights on misclassified samples.
5. **1D Convolutional Neural Network (1D-CNN):**
   - **Why 1D-CNN on tabular data?** Treats the 18 scaled features as an ordered 1D sequence tensor of shape `(18, 1)`. 1D convolutional filter banks extract local sub-patterns and feature interaction motifs.
   - **Architecture:**
     - Layer 1: `Input(shape=(18, 1))`
     - Layer 2: `Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')`
     - Layer 3: `MaxPooling1D(pool_size=2)`
     - Layer 4: `Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')`
     - Layer 5: `Flatten()`
     - Layer 6: `Dense(64, activation='relu')`
     - Layer 7: `Dropout(0.30)` (regularization to avoid co-adaptation)
     - Layer 8: `Dense(1, activation='sigmoid')`
   - **Optimizer:** Adam ($\text{lr} = 0.001$).
   - **Loss Function:** Binary Crossentropy with class-weight multipliers.
   - **Regularization:** `EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)`.

---

## 7. Explainable AI (XAI) via SHAP

### Why SHAP?
Deep neural nets and gradient boosted trees are frequently criticized as "black boxes" in healthcare. SHAP (SHapley Additive exPlanations) grounds feature importance in cooperative game theory (Lloyd Shapley, Nobel Laureate).

### The Math:
The prediction $f(x)$ is decomposed as:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$
Where $\phi_0$ is the base expected model output and $\phi_i$ is the exact marginal contribution of feature $i$.

### Top 5 Global Features Discovered:
1. **`Age` (Mean \|SHAP\| = 3.2672):** Strong non-linear inflection in young adults (18-24) and late career workers.
2. **`Suicidal Thoughts = Yes` (Mean \|SHAP\| = 1.3009):** The strongest psychiatric risk indicator.
3. **`Work Pressure` (Mean \|SHAP\| = 1.2770):** Levels 4 and 5 heavily push predictions toward positive depression status.
4. **`Job Satisfaction` (Mean \|SHAP\| = 1.2440):** Low satisfaction (<2) acts as a compounding vulnerability.
5. **`Work Hours` (Mean \|SHAP\| = 1.0417):** Excessive hours (>10h/day) compound sleep deficit.

---

## 8. Master Experimental Results Benchmark

*Evaluated on independent stratified test set ($N = 411$, containing 41 positive cases and 370 negative cases).*

| Approach / Subset | Model Architecture | Accuracy | Balanced Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep Learning** | **1D-CNN (Keras)** | **98.54%** | **99.19%** | **87.23%** | **100.00%** | **93.18%** | **0.9989** |
| All Features (Baseline) | XGBoost | 98.05% | 93.50% | 92.31% | 87.80% | 90.00% | 0.9940 |
| All Features (Baseline) | SVM (RBF Balanced) | 97.32% | 96.34% | 81.25% | 95.12% | 87.64% | 0.9963 |
| PSO Feature Selection | XGBoost | 96.84% | 91.74% | 83.33% | 85.37% | 84.34% | 0.9920 |
| PSO Feature Selection | SVM | 95.86% | 94.45% | 73.08% | 92.68% | 81.72% | 0.9906 |
| A* Feature Selection | XGBoost | 96.11% | 90.25% | 79.07% | 82.93% | 80.95% | 0.9891 |
| PSO Feature Selection | AdaBoost | 96.59% | 84.01% | 96.55% | 68.29% | 80.00% | 0.9913 |
| All Features (Baseline) | AdaBoost | 96.35% | 81.71% | 100.00% | 63.41% | 77.61% | 0.9940 |
| A* Feature Selection | SVM | 94.65% | 93.77% | 66.67% | 92.68% | 77.55% | 0.9904 |
| A* Feature Selection | AdaBoost | 96.11% | 82.66% | 93.10% | 65.85% | 77.14% | 0.9879 |
| A* Feature Selection | Random Forest | 95.38% | 77.91% | 95.83% | 56.10% | 70.77% | 0.9842 |
| PSO Feature Selection | Random Forest | 95.38% | 76.83% | 100.00% | 53.66% | 69.84% | 0.9864 |
| All Features (Baseline) | Random Forest | 94.16% | 70.73% | 100.00% | 41.46% | 58.62% | 0.9885 |

---

## 9. Direct Answers to Synopsis Research Questions

### Research Question 1 (RQ1):
**Question:** *Which demographic, occupational, behavioral, and family-history features contribute most to depression prediction?*  
**Empirical Answer:**  
Through SHAP global attribution and single-feature F1 evaluation, the top predictive attributes are:
1. **Psychiatric Alert:** History of Suicidal Thoughts (strongest categorical predictor).
2. **Occupational Factors:** Work Pressure (Level 4-5) combined with Work Hours (>10h) and Low Job Satisfaction.
3. **Biological & Behavioral:** Sleep Duration (<5 hours deficit) and Unhealthy Diet.
4. **Familial/Genetic:** Family History of Mental Illness.

---

### Research Question 2 (RQ2):
**Question:** *Which classification approach provides the most effective predictive performance?*  
**Empirical Answer:**  
The **1D Convolutional Neural Network (CNN)** achieved the highest predictive performance:
- **Accuracy:** 98.54%
- **Recall (Sensitivity):** **100.00%** (Identified all 41 positive cases on test data)
- **F1-Score:** 93.18%
- **ROC-AUC:** 0.9989  
Among classical models, **XGBoost** achieved the highest F1-score (90.00%), while **SVM** with balanced class weights achieved the highest classical recall (95.12%).

---

### Research Question 3 (RQ3):
**Question:** *Whether optimization and explainability techniques can improve feature selection and interpretation of model predictions?*  
**Empirical Answer:**  
- **Optimization:** Both A* and PSO successfully compressed the feature space from 18 features down to **10 features** (a 44.4% dimensional reduction). A* maintained a validation F1 of 91.18% and PSO achieved 88.89%, proving that lighter, low-dimensional feature subsets can preserve high clinical screening power.
- **Explainability:** SHAP transformed the "black box" models into transparent decision pathways, providing both global factor ranking and localized patient-level explanations.

---

## 10. MindCare AI Web Application Architecture

To translate theoretical models into practical healthcare technology, we developed **MindCare AI**:
- **Backend:** Python + Flask (`app.py`), loading serialized artifacts (`preprocessor.joblib`, `svm_model.joblib`, `xgboost_model.joblib`, `depression_cnn_model.keras`).
- **Endpoints:**
  - `POST /api/predict`: Runs real-time inference on 10 user inputs, calculating calibrated risk percentages and identifying individual driver factors.
  - `GET /api/metadata`: Serves benchmark tables, A*/PSO feature lists, and formal research answers.
- **Frontend:** Semantic HTML5 + Disciplined Clinical Vanilla CSS + Vanilla JS.
- **Features:**
  - Live interactive sliders (Age, Work Pressure, Job Satisfaction, Work Hours, Financial Stress).
  - Model selector allowing comparison between 1D-CNN, XGBoost, and SVM.
  - Three pre-configured test profiles (Low Risk, Occupational Strain, Clinical Alert).
  - Clean shimmer Skeleton Loader during API inference.
  - Terms of Service & Data Privacy Policy modals explaining academic boundaries and medical disclaimers.

---

## 11. Comprehensive Viva & Oral Defense Questions

Below are 15 targeted questions frequently asked by university project evaluators and professors, paired with clear, professional answers.

---

### Q1: Why did you choose Machine Learning for depression prediction instead of standard psychological questionnaires?
**Model Answer:**  
"Traditional psychiatric screening tools like PHQ-9 or BDI rely on self-reported feelings after symptoms have already developed. Our machine learning approach uses objective, measurable lifestyle and occupational attributes—such as daily work hours, financial stress level, sleep duration, and job satisfaction. This allows for upstream, proactive risk stratification in corporate and healthcare settings before severe clinical manifestations occur."

---

### Q2: Why is raw Accuracy a misleading metric in your project?
**Model Answer:**  
"Because of the severe 90.12% to 9.88% class imbalance in our dataset. A naive model predicting 'No Depression' for all 2,054 records would automatically achieve 90.12% accuracy while completely failing to detect a single depressed individual. In medical diagnostics, a False Negative (missing a depressed patient) carries catastrophic risk. Therefore, we prioritized **Recall (Sensitivity)** and the **F1-Score** over raw Accuracy."

---

### Q3: How did you handle the class imbalance during model training?
**Model Answer:**  
"We applied a multi-tiered strategy:
1. **Stratification:** We used `stratify=y` during the train/test split to preserve the 9:1 ratio in both sets.
2. **Cost-Sensitive Learning (`scale_pos_weight`):** In XGBoost, we set the positive penalty weight to $\approx 9.14$ (negative/positive ratio).
3. **Balanced Class Weighting:** In SVM and Random Forest, class weights inversely proportional to class frequencies were passed to adjust the loss gradient.
4. **Neural Network Class Weighting:** In the 1D-CNN, loss weights penalized false negatives during backpropagation."

---

### Q4: Explain the difference between A* Search and Particle Swarm Optimization (PSO) in your feature selection.
**Model Answer:**  
"A* is a **deterministic, heuristic graph search** algorithm. It maintains an open queue of states sorted by $f(n) = g(n) + h(n)$, evaluating the exact validation F1-score plus a length penalty, guided by an optimistic heuristic of remaining feature potentials.  
In contrast, PSO is a **stochastic, population-based swarm intelligence** metaheuristic. A swarm of particles moves through an 18-dimensional discrete binary space, updating their velocity vectors based on their personal best memory ($p_{\text{best}}$) and global swarm best ($g_{\text{best}}$). A* guarantees heuristic optimality within evaluated bounds, while PSO excels at rapid exploration across high-dimensional non-convex search spaces."

---

### Q5: Why did you choose a 1D Convolutional Neural Network (CNN) for tabular data instead of just traditional ML?
**Model Answer:**  
"While CNNs are famously used for 2D images, 1D-CNNs are effective for structured tabular vectors when normalized and treated as ordered 1D sequence tensors of shape $(18, 1)$. The 1D convolutional kernels ($3 \times 1$) slide across adjacent attribute sets, capturing local cross-feature relationships and abstract interaction patterns. Combined with pooling and dropout regularization, our 1D-CNN achieved 100.00% recall on the minority cohort."

---

### Q6: How do you prevent Data Leakage during preprocessing?
**Model Answer:**  
"Data leakage occurs when information from the test dataset influences training transformations. To eliminate leakage, we performed the `train_test_split` first. The `StandardScaler` and `OneHotEncoder` within our `ColumnTransformer` were fitted **strictly on `X_train`** using `.fit_transform()`. The test set `X_test` was only transformed using `.transform()` with the pre-learned training parameters."

---

### Q7: What is SHAP, and how does it explain model decisions?
**Model Answer:**  
"SHAP stands for SHapley Additive exPlanations, grounded in cooperative game theory. It calculates the unique marginal contribution of each feature across all possible feature combinations. Unlike simple Gini importance in tree ensembles (which only tells you that a feature was used frequently), SHAP values explain whether a feature increased or decreased the risk probability for an individual patient, ensuring clinical interpretability."

---

### Q8: What were the most influential features identified by SHAP?
**Model Answer:**  
"The top 5 features were:
1. `Age` (non-linear risk distribution across age cohorts).
2. `Suicidal Thoughts = Yes` (strongest individual risk driver).
3. `Work Pressure` (severe pressure $\ge 4$ significantly increases positive log-odds).
4. `Job Satisfaction` (low satisfaction compounds stress).
5. `Daily Work Hours` (>10 hours/day).  
These findings validate that occupational burnout and severe psychological strain are closely intertwined with clinical depression."

---

### Q9: Why did Random Forest have a lower Recall (41.46%) compared to SVM (95.12%) and 1D-CNN (100.00%)?
**Model Answer:**  
"Decision trees split nodes based on Gini impurity. In heavily imbalanced datasets (90:10), individual leaf nodes are dominated by majority negative samples. Even with `class_weight='balanced'`, uncalibrated decision thresholds at the default $0.5$ probability mark can under-predict the rare class. In contrast, SVM with an RBF kernel constructs a non-linear maximum-margin boundary that penalizes minority misclassifications heavily, and 1D-CNN dynamically adjusts its gradient vectors via backpropagation."

---

### Q10: How does your web application MindCare AI work under the hood?
**Model Answer:**  
"MindCare AI is built on a Flask REST API backend. When a clinician or user inputs parameters:
1. The JSON payload is validated and organized into a DataFrame matching the training schema.
2. The serialized `preprocessor.joblib` pipeline scales and encodes the inputs.
3. The selected model (`depression_cnn_model.keras`, `xgboost_model.joblib`, or `svm_model.joblib`) performs live inference.
4. The API returns the calculated probability, risk classification (Low, Moderate, High), and tailored recommendations based on the highest contributing stress factors."

---

### Q11: What is Early Stopping in your 1D-CNN, and why did you use it?
**Model Answer:**  
"Early Stopping is a regularization callback that monitors validation loss (`val_loss`) during training epochs. If `val_loss` fails to improve for 8 consecutive epochs (patience = 8), training terminates automatically, and model weights are restored to the best-performing epoch. This prevents the neural network from memorizing the training noise and overfitting."

---

### Q12: Did feature selection improve or hurt performance?
**Model Answer:**  
"Feature selection compressed the feature space from 18 features down to 10 features—a **44.4% reduction in dimensionality**.  
Performance was preserved: A* selected features achieved an F1-score of 91.18% on validation data, and XGBoost trained on PSO features retained an F1 of 84.34% and ROC-AUC of 0.9920 on test data. This proves that high diagnostic performance can be achieved with a lighter, faster feature set."

---

### Q13: What are the ethical and clinical limitations of this project?
**Model Answer:**  
"1. **Ethical Disclaimer:** This is an academic decision-support tool, not a certified medical device. It cannot provide formal psychiatric diagnoses.
2. **Subjectivity:** Self-reported survey data can introduce social desirability bias.
3. **Cross-Sectional Dataset:** The dataset captures a single snapshot in time. Longitudinal tracking over multiple years would provide greater temporal insight."

---

### Q14: What would you do as Future Work for a Major Project?
**Model Answer:**  
"For our Major Project, we aim to:
1. Validate on larger, multi-hospital clinical cohorts.
2. Integrate continuous physiological telemetry from consumer wearables (sleep stage architecture, Heart Rate Variability - HRV).
3. Explore hybrid optimization algorithms, such as combining Genetic Algorithms with PSO (GA-PSO)."

---

### Q15: How is your project reproducible?
**Model Answer:**  
"All random states were pinned (`random_state=42`, `tf.random.set_seed(42)`). The preprocessor and all trained models are serialized using `joblib` and Keras `.keras` formats. Running `python depression_prediction.py` replicates the exact numbers, tables, and figures."

---

## 12. Group Member Presentation Strategy

| Member Name | Presentation Topic | Key Technical Talking Points |
| :--- | :--- | :--- |
| **Piyush Raj** | Problem Definition, Synopsis Objectives & Preprocessing | Explains the 90:10 class imbalance challenge, dataset variables, `StandardScaler`, `OneHotEncoder`, and data leakage prevention. |
| **Ipsita Gayatri** | Optimization Algorithms (A* Search & Binary PSO) | Explains heuristic cost $f(n) = g(n) + h(n)$ in A*, particle velocity equations in PSO, and how 18 features were reduced to 10. |
| **Mohit Kumar** | Model Architectures & Deep Learning (1D-CNN) | Explains the 5 models, why 1D-CNN outperformed classical models (100% recall), Conv1D layers, dropout, and early stopping. |
| **Rajesh Kumar Mahato** | Explainable AI (SHAP), Web Prototype & Answers to RQs | Explains Shapley values, top 5 risk drivers, demonstrates the live MindCare AI web application, and delivers answers to RQ1, RQ2, RQ3. |

---
*Handbook prepared for Group 22 Minor Project MO/26 Evaluation at BIT Mesra Off-Campus Deoghar.*
