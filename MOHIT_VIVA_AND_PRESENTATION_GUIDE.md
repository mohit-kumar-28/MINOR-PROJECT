# MindCare AI : Mohit Kumar's Dedicated Viva & Presentation Guide
**Student Name:** Mohit Kumar  
**Roll / Reg No:** BTECH/60116/23  
**Project Title:** Predicting Depression and Mental Health Issues  
**Group No:** 22 | Semester: 7th (Minor Project MO/26)  
**Assigned Domain:** **Machine Learning & Deep Learning Architectures, Model Training, 1D-CNN, & Performance Benchmarks**  

---

## Table of Contents
1. [Mohit Ka Role & Summary (Aapko Kya Bolna Aur Samjhana Hai)](#1-mohit-ka-role--summary)
2. [Exact Presentation Script (Word-by-Word English Script for Slides)](#2-exact-presentation-script-for-slides)
3. [Deep-Dive: The 5 Model Architectures Explained](#3-deep-dive-the-5-model-architectures-explained)
4. [Master Technical Focus: 1D Convolutional Neural Network (CNN)](#4-master-technical-focus-1d-convolutional-neural-network-cnn)
5. [The Benchmark Results Table (Aapka Main Result Slide)](#5-the-benchmark-results-table)
6. [Why Models Behaved This Way (Key Analytical Insights)](#6-why-models-behaved-this-way)
7. [Code Walkthrough of Mohit's Section (`depression_prediction.py`)](#7-code-walkthrough-of-mohits-section)
8. [Top 12 Viva & Cross-Questions for Mohit (With Model Answers)](#8-top-12-viva--cross-questions-for-mohit)

---

## 1. Mohit Ka Role & Summary

Mohit, aapka main role viva aur presentation mein **Models, Deep Learning (1D-CNN), aur Benchmark Results** ko lead karna hai.

Jab aapke group members problem statement (Piyush) aur feature selection (Ipsita) cover kar lenge, tab aapka turn aayega **Slide 6 aur Slide 7** par.

### Aapko 3 Main Cheezein Explain Karni Hain:
1. **Humne 5 models kyu choose kiye:** SVM, Random Forest, XGBoost, AdaBoost, aur 1D Convolutional Neural Network (CNN).
2. **1D-CNN kaise kaam karta hai tabular data par:** Scaled 18 features ko sequential tensor $(18, 1)$ mein daal kar local feature relationships extract karna.
3. **Results aur Comparisons:** 1D-CNN ne **98.54% Accuracy, 100.00% Recall, aur 93.18% F1-Score** ke sath sabse best performance kyu di, aur Random Forest ka Recall kyu drop hua.

---

## 2. Exact Presentation Script for Slides

*(Jab presentation mein aapka turn aaye, aap confident hokar yeh English script bol sakte hain. Neeche brackets mein Hindi context bhi diya hai taaki aap ratta na marein, balki natural lage).*

---

### Step 1: Handover Lene Ka Script
> *"Thank you, Ipsita. Good morning/afternoon respected supervisor and coordinator. I am Mohit Kumar (BTECH/60116/23). I will be presenting the Classification Framework, Deep Learning Architecture, and the Comparative Performance Benchmarks of our study."*

---

### Step 2: Slide 6 (Proposed Methodology: Classification & Evaluation)
> *"Moving to our classification methodology, rather than relying on a single family of algorithms, we benchmarked five distinct classifiers spanning linear boundary mapping, bagging ensembles, gradient boosting, and deep neural networks:*
> 1. *Support Vector Machines with an RBF kernel and balanced class weighting.*
> 2. *Random Forest with 200 de-correlated bootstrap decision trees.*
> 3. *XGBoost utilizing second-order gradient boosting and scale_pos_weight of 9.14.*
> 4. *AdaBoost focusing on adaptive sample re-weighting.*
> 5. *And a custom 1D Convolutional Neural Network designed for tabular sequence processing.*
>
> *Each model was trained on the common baseline feature set, as well as the optimized subsets selected by A\* and PSO, and evaluated on the same held-out test split of 411 instances."*

---

### Step 3: Slide 7 (Results and Discussion: Table & Insights)
> *"Now looking at Slide 7, here are our empirical findings across the benchmark:*
>
> *First, regarding our class imbalance—where depression-positive cases represent approximately 9.9%—raw accuracy alone is insufficient. We placed primary emphasis on Recall (Sensitivity) and the F1-Score.*
>
> *As shown in the benchmark table:*
> - *Our **1D-CNN achieved superior overall performance**, reaching an **Accuracy of 98.54%**, a **Recall of 100.00%** on the minority class, an **F1-Score of 93.18%**, and an **ROC-AUC of 0.9989**.*
> - *Among traditional classifiers, **XGBoost achieved the highest F1-Score of 90.00%** with an accuracy of 98.05%.*
> - *The **SVM model achieved strong recall at 95.12%**, demonstrating that cost-sensitive margin boundary adjustment successfully shielded against majority-class dominance.*
> - *Random Forest achieved 94.16% accuracy, but exhibited lower recall at 41.46%, because individual decision trees split primarily on dominant negative samples unless threshold calibration is applied.*
>
> *Furthermore, models trained on the 10 features selected by A\* and PSO retained competitive F1-scores while cutting feature dimensionality by over 44%. I will now hand over to Rajesh to discuss Explainable AI with SHAP and our project conclusions."*

---

## 3. Deep-Dive: The 5 Model Architectures Explained

### 1. Support Vector Machine (SVM)
- **Concept:** Finds an optimal hyperplane that maximizes the margin between depressed and non-depressed points.
- **Kernel:** Radial Basis Function (RBF) $K(x, x') = \exp(-\gamma ||x - x'||^2)$, which projects non-linear feature interactions into higher-dimensional space.
- **Imbalance Handling:** `class_weight='balanced'`. Calculates class penalty $w_j = \frac{N}{2 \times N_j}$. The minority class receives a $\approx 5\times$ higher misclassification penalty.
- **Performance:** **Accuracy: 97.32% | Recall: 95.12% | F1: 87.64% | AUC: 0.9963**.

---

### 2. Random Forest Classifier
- **Concept:** Ensemble Bagging (Bootstrap Aggregation) technique.
- **Hyperparameters:** `n_estimators=200`, `class_weight='balanced'`.
- **How it works:** 200 trees are trained on random subsets of rows and features. Predictions are averaged by majority vote.
- **Why its recall was 41.46%:** In an imbalanced 90:10 dataset, individual tree leaf nodes are overwhelmingly dominated by non-depressed samples. Even with balanced weighting, the default 0.5 decision threshold causes it to miss subtle positive cases, leading to high precision (100%) but moderate recall.

---

### 3. XGBoost (Extreme Gradient Boosting)
- **Concept:** Sequential boosting where each new decision tree minimizes the residual errors of prior trees using a second-order Taylor expansion.
- **Hyperparameters:** `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `scale_pos_weight=9.14`.
- **Why scale_pos_weight is key:** It multiplies the loss gradient for positive samples by 9.14, directly correcting for the 1851:203 imbalance.
- **Performance:** **Accuracy: 98.05% | Recall: 87.80% | F1: 90.00% | AUC: 0.9940**. (Best classical model overall).

---

### 4. AdaBoost (Adaptive Boosting)
- **Concept:** Focuses sequentially on "hard-to-classify" samples. After each weak learner (stump) is trained, sample weights for misclassified points are boosted.
- **Hyperparameters:** `n_estimators=100`, `learning_rate=0.5`.
- **Performance:** **Accuracy: 96.35% | Recall: 63.41% | F1: 77.61% | AUC: 0.9940**.

---

### 5. 1D Convolutional Neural Network (CNN)
*(See Section 4 for deep technical architectural breakdown).*

---

## 4. Master Technical Focus: 1D Convolutional Neural Network (CNN)

Evaluators frequently ask Mohit: **"CNN toh images ke liye hota hai, aapne tabular data mein CNN kyu use kiya?"**

Here is the complete technical explanation:

### Why 1D-CNN on Tabular Data?
1. **Input Reshape:** The 18 preprocessed continuous/one-hot features are reshaped into a 3D tensor:  
   $$\text{Shape: } (N, \text{features}, \text{channels}) = (N, 18, 1)$$
2. **Local Feature Interactions:** A 1D convolutional kernel of size $3 \times 1$ slides over 3 adjacent features at a time. It calculates dot-products and discovers complex non-linear feature conjunctions (e.g., *Age + Work Pressure + Job Satisfaction*).
3. **Weight Sharing:** Unlike an MLP where every input neuron has independent dense weights, convolutional filters share weights across the feature vector, reducing parameter count and preventing overfitting.

### The Exact Layer Architecture:
```
Input Layer: Tensor shape (18, 1)
   │
   ▼
Conv1D Layer 1: 32 Filters, Kernel Size = 3, Activation = 'ReLU', Padding = 'same'
   │ Output Shape: (18, 32)
   ▼
MaxPooling1D Layer: Pool Size = 2
   │ Output Shape: (9, 32) - Halves dimensionality, extracts dominant signals
   ▼
Conv1D Layer 2: 64 Filters, Kernel Size = 3, Activation = 'ReLU', Padding = 'same'
   │ Output Shape: (9, 64) - Higher-level abstract feature interactions
   ▼
Flatten Layer: Converts 2D feature maps to 1D vector of 576 nodes (9 × 64)
   │
   ▼
Dense Layer: 64 Neurons, Activation = 'ReLU'
   │ Fully connected classification layer
   ▼
Dropout Layer: Rate = 0.30 (30% neurons randomly dropped during training)
   │ Prevents co-adaptation and overfitting
   ▼
Output Layer: 1 Neuron, Activation = 'Sigmoid'
   │ Outputs calibrated probability P(Depression = 1) between 0.0 and 1.0
```

### Training & Regularization Settings:
- **Optimizer:** Adam ($\alpha = 0.001$).
- **Loss Function:** Binary Crossentropy:
  $$\mathcal{L} = - [w_1 \cdot y \log(\hat{y}) + w_0 \cdot (1-y) \log(1 - \hat{y})]$$
- **Class Weights:** Passed $w_1 = 9.14$ and $w_0 = 1.0$ to force the network to aggressively learn minority features.
- **Early Stopping:** `patience=8`, monitoring `val_loss`. Stopped early at epoch 18 to restore the best weights and prevent overfitting.

---

## 5. The Benchmark Results Table

*Yeh table aapko yaad ya clear honi chahiye kyunki evaluate karte waqt numbers pooche jaate hain:*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Key Advantage / Trade-off |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1D-CNN (Keras)** | **98.54%** | **87.23%** | **100.00%** | **93.18%** | **0.9989** | **Top Overall.** Zero false negatives on test set. |
| **XGBoost (Baseline)** | **98.05%** | **92.31%** | **87.80%** | **90.00%** | **0.9940** | **Best Classical.** High precision & strong F1. |
| **SVM (RBF Balanced)** | **97.32%** | **81.25%** | **95.12%** | **87.64%** | **0.9963** | **High Sensitivity.** Catches 39 out of 41 cases. |
| **AdaBoost** | **96.35%** | **100.00%** | **63.41%** | **77.61%** | **0.9940** | **Perfect Precision.** Zero false alarms, but missed 15 cases. |
| **Random Forest** | **94.16%** | **100.00%** | **41.46%** | **58.62%** | **0.9885** | **Conservative.** Misses minority class at 0.5 threshold. |

---

## 6. Why Models Behaved This Way

Professors often test if you understand the *underlying theory* behind these numbers. Here is the exact reasoning:

### 1. Why 1D-CNN achieved 100% Recall:
- The combination of **class-weighted binary crossentropy loss** and **1D spatial convolutions** allowed the neural net to map subtle cross-feature interactions (such as someone with moderate work pressure but severe sleep deficit and suicidal thoughts) that linear or shallow models overlook.

### 2. Why SVM beat Random Forest on Recall (95.12% vs 41.46%):
- Random Forest splits nodes based on pure sample counts (Gini impurity). When 90% of samples are "No", trees naturally favor the majority.
- SVM optimizes the margin *support vectors* specifically. Because `class_weight='balanced'` penalizes support vectors from the minority class heavily, it pushes the decision boundary safely away from the depressed cluster, capturing almost all positive cases.

### 3. Why XGBoost had the highest Precision (92.31%) among top models:
- XGBoost uses regularized boosting ($L_1$ and $L_2$ tree complexity penalties). It doesn't over-predict positives haphazardly; it only creates positive branches when loss reduction is statistically significant.

---

## 7. Code Walkthrough of Mohit's Section

In [`depression_prediction.py`](file:///c:/Users/BIT/Desktop/boa/depression_prediction.py), Mohit's logic is implemented in **Step 7, Step 12, and Step 15**:

### Step 7: Baseline Model Definitions (Lines 200–260)
```python
def get_models_dict():
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, # 9.14 class penalty
        eval_metric="logloss",
        random_state=42
    )
    return {
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "XGBoost": xgb,
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
    }
```

### Step 12: 1D-CNN Model Architecture (Lines 380–435)
```python
cnn_model = Sequential([
    Input(shape=(n_features, 1)),
    Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.30),
    Dense(1, activation="sigmoid")
])

cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

# Stratified Validation Split & Class-Weighted Training
cw = {0: 1.0, 1: scale_pos_weight}
cnn_model.fit(
    X_tr_c, y_tr_c,
    validation_data=(X_val_c, y_val_c),
    epochs=40,
    batch_size=32,
    class_weight=cw,
    callbacks=[early_stop]
)
```

---

## 8. Top 12 Viva & Cross-Questions for Mohit

Yeh 12 questions specifically Mohit ke part ke upar banaye gaye hain. Evaluators usually yahi sawaal puchenge:

---

### Q1: Mohit, why did you choose a 1D-CNN instead of a simple Multi-Layer Perceptron (MLP/ANN)?
**Answer to speak:**  
*"Sir/Ma'am, in a standard MLP, every neuron is fully connected to all input features, which creates a huge number of independent parameters and makes the model prone to overfitting on smaller tabular datasets.  
A 1D-CNN uses **parameter sharing** and **local receptive fields**. The convolutional filters slide over adjacent features in the vector, capturing localized interaction motifs (for example, the joint interaction between Work Hours, Work Pressure, and Sleep Duration) while keeping parameter counts low. Furthermore, pooling operations extract the most dominant signals."*

---

### Q2: What is the purpose of the `kernel_size=3` in your Conv1D layer?
**Answer to speak:**  
*"The `kernel_size=3` specifies the receptive field width of the sliding 1D filter. It means the filter evaluates triplets of neighboring preprocessed features simultaneously at each step, computing a localized weighted sum to detect interaction patterns."*

---

### Q3: Your 1D-CNN achieved 100% Recall. Is it possible that the model is overfitting?
**Answer to speak:**  
*"That was an important question we verified. It is not overfitting because:  
1. We evaluated the 100% recall strictly on an **unseen held-out test split of 411 records**, which was completely isolated during training.  
2. The model achieved a high **ROC-AUC of 0.9989** and an **Accuracy of 98.54%** on that same test set, with only a small number of False Positives.  
3. We integrated **Dropout (0.30)** and **Early Stopping with patience=8**, which actively halted training when validation loss began to plateau, preventing weight memorization."*

---

### Q4: What is the difference between Bagging (Random Forest) and Boosting (XGBoost/AdaBoost)?
**Answer to speak:**  
*"Both are ensemble methods, but their training philosophy differs:  
- **Bagging (Random Forest):** Trains 200 decision trees **in parallel** on random bootstrap subsets of data and features. It reduces **variance** by averaging independent tree outputs.  
- **Boosting (XGBoost & AdaBoost):** Trains decision trees **sequentially**. Each new tree focuses specifically on correcting the errors and residuals left by preceding trees, primarily reducing **bias**."*

---

### Q5: How did you calculate `scale_pos_weight` in XGBoost?
**Answer to speak:**  
*"We calculated `scale_pos_weight` as the ratio of negative majority instances to positive minority instances in the training set:  
$$\text{scale\_pos\_weight} = \frac{\text{Count}(y=0)}{\text{Count}(y=1)} = \frac{1481}{162} \approx 9.14$$  
In the binary logistic loss function, this multiplies the gradient and hessian of positive instances by 9.14, heavily penalizing false negatives."*

---

### Q6: What kernel did you use in SVM and why?
**Answer to speak:**  
*"We used the **Radial Basis Function (RBF)** kernel. Mental health attributes have complex, non-linear relationships with depression risk. A simple linear kernel would assume a straight boundary line in feature space. The RBF kernel maps features into an infinite-dimensional space using Gaussian similarity, allowing it to construct curved, non-linear separation boundaries."*

---

### Q7: Why did Random Forest give 100% Precision but only 41.46% Recall?
**Answer to speak:**  
*"In Random Forest, when trees vote on an imbalanced dataset, the default probability threshold is $0.5$. Because negative samples dominate the dataset, tree leaves rarely accumulate enough positive votes to surpass 0.5 unless the positive signal is overwhelmingly obvious. Hence, every sample it predicted positive was indeed positive (100% precision), but it missed more subtle positive cases (41.46% recall)."*

---

### Q8: What does `patience=8` mean in Early Stopping?
**Answer to speak:**  
*"Patience represents the epoch tolerance window. If the validation loss (`val_loss`) fails to achieve a new minimum for 8 consecutive training epochs, Keras halts training immediately and restores the model weights from the best epoch via `restore_best_weights=True`."*

---

### Q9: Why did you use the Sigmoid activation function at the output of the CNN rather than Softmax?
**Answer to speak:**  
*"Because our problem is a **binary classification problem** (Depression: Yes or No). The Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ squashes any real value into a smooth scalar probability range between $0.0$ and $1.0$. Softmax is typically reserved for multi-class classification where probabilities across multiple discrete classes must sum to 1."*

---

### Q10: What is the difference between ROC-AUC and Precision-Recall (PR) AUC?
**Answer to speak:**  
*"ROC-AUC plots True Positive Rate versus False Positive Rate. It provides a measure of overall ranking ability across all thresholds.  
However, on heavily imbalanced datasets, PR-AUC is often more rigorous because it plots Precision against Recall, focusing specifically on the minority positive class without being inflated by a large true-negative count."*

---

### Q11: How did you evaluate the models on A* and PSO feature subsets?
**Answer to speak:**  
*"After A* and PSO selected their 10 optimal feature indices from the 18 available features, we subsetted the processed training matrix $X_{\text{train}}$ and test matrix $X_{\text{test}}$ to contain only those 10 columns. We then re-trained all 4 classical models from scratch on these reduced subsets and evaluated them against the identical test labels."*

---

### Q12: In our web app MindCare AI, which model is set as the default predictor?
**Answer to speak:**  
*"Our deployed web app sets the **1D-CNN** as the default recommended model because of its 100% sensitivity on the minority cohort. However, we also built a dropdown selector enabling clinicians to toggle and compare predictions from XGBoost and SVM in real-time."*

---
*Guide prepared for Mohit Kumar (BTECH/60116/23) - Group 22 Minor Project MO/26 Evaluation at BIT Mesra Off-Campus Deoghar.*
