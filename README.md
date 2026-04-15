# Wearable-Stress-Activity-Classification ⌚🧠🏃‍♂️

> A robust Machine Learning pipeline to classify human physiological states (Rest, Cognitive Stress, Physical Activity) using biometric data from the Empatica E4 wearable sensor.

## 📌 Project Overview
The goal of this project is to accurately distinguish between three physiological states using non-invasive wearable data. While physical activity is easily detectable, the true challenge lies in differentiating **cognitive stress** from a **resting state**, as both involve minimal physical movement but distinct autonomic nervous system responses.

This project implements a full end-to-end Machine Learning pipeline, from raw signal processing and feature extraction to hyperparameter optimization and model interpretability.

**Classes predicted:**
* `0`: Rest / Baseline
* `1`: Cognitive Stress
* `2`: Physical Activity

## 📊 Dataset
The data used in this project was collected using the **Empatica E4** wristband, which records:
* **ACC:** 3-axis Accelerometer
* **BVP:** Blood Volume Pulse (used to derive HRV)
* **EDA:** Electrodermal Activity (Galvanic Skin Response)
* **TEMP:** Skin Temperature
* **HR:** Heart Rate

🔗 **Dataset Source:** [[Wearable Device Dataset](https://physionet.org/content/wearable-device-dataset/1.0.1/Wearable_Dataset/)]  
*Please refer to the original authors for full details on the experimental protocol and data collection methodology.*

## ⚙️ Methodology & Pipeline

### 1. Feature Extraction (`flirt`)
Used the `flirt` (Feature Generation Toolkit for Wearable Data) library to extract advanced statistical, time-domain, and frequency-domain features from raw signals using rolling windows.

### 2. Data Quality & EDA
* **Anomaly Detection:** Identified and handled mathematical anomalies (e.g., `-inf` values caused by $\log(0)$ in entropy calculations during zero-movement windows).
* **Zero-Variance Removal:** Dropped static features with no predictive power.
* **Dimensionality Reduction:** Eliminated highly correlated redundant features (Pearson correlation > 0.85) to prevent multicollinearity.

### 3. Preventing Data Leakage
Instead of a standard random split, **`GroupShuffleSplit`** was implemented using `Patient_ID`. This ensures that the model is evaluated on *completely unseen patients*, proving its real-world generalization capabilities rather than just memorizing specific baselines.

### 4. Handling Class Imbalance
The dataset naturally contained more 'Rest' segments than 'Stress' segments. To prevent the model from becoming biased towards the majority class:
* Models were trained with `class_weight='balanced'`.
* `GridSearchCV` was optimized for the **`f1_macro`** metric rather than simple accuracy.

### 5. Modeling
Multiple models were evaluated and tuned via Grid Search, including:
* Random Forest (Winner 🏆)
* XGBoost
* Support Vector Machines (SVM)
* Logistic Regression
* Multi-layer Perceptron (MLP)
* K-Nearest Neighbors (KNN)

## 📈 Results & Evaluation

The **Random Forest Classifier** achieved the best balance between precision and recall, successfully learning the subtle physiological differences between Stress and Rest while perfectly isolating Physical Activity.

### Confusion Matrix
*(The model showed a high recall for stress detection after fixing class imbalance).*
![Confusion Matrix](images/confussion_matrix.png)

### Multiclass ROC-AUC (One-vs-Rest)
*(The AUC score for Physical Activity was nearly perfect, while Stress and Rest showed strong, robust separation).*
![ROC AUC Curve](images/AUC_ROC.png)

### Feature Importance
Interpretability analysis revealed that the model correctly relied on physiological markers (like EDA peaks and HRV metrics) to detect stress, rather than relying solely on accelerometer data.
![Feature Importance](images/feature_importance.png)

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone [https://github.com/YourUsername/Wearable-Stress-Activity-Classification.git](https://github.com/YourUsername/Wearable-Stress-Activity-Classification.git)
   
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

3. Download the dataset from the link provided above and place the AEROBIC, ANAEROBIC and STRESS folders inside the data/ directory.

4. Run the Jupyter Notebooks sequentially to replicate the extraction, EDA, and training processes.

## 🤝 Acknowledgements
Special thanks to the creators of the flirt library and the researchers who open-sourced the Empatica E4 dataset.
