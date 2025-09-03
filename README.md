# 🧩 End-to-End ML Pipeline with DVC & MLflow

This project demonstrates how to build a **reproducible machine learning pipeline** on the **Pima Indians Diabetes Dataset**, combining:

* 🔹 **DVC** for dataset & model versioning
* 🔹 **MLflow** for experiment tracking
* 🔹 **Scikit-learn** for model building (Random Forest Classifier)
* 🔹 **DagsHub** for remote data & experiment collaboration

The goal is to showcase **MLOps best practices** from data preparation to experiment tracking and deployment readiness.

---

## 🚀 Features

* ✅ Preprocess raw dataset into clean training data
* ✅ Train a **Random Forest Classifier** on preprocessed data
* ✅ Log metrics, parameters, and artifacts to **MLflow**
* ✅ Track & version datasets, models, and pipelines using **DVC**
* ✅ Reproduce experiments with a single command
* ✅ Store & share experiments and artifacts on **DagsHub**
* ✅ Collaborate seamlessly with remote storage + MLflow UI

---

## ⚙️ Technologies Used

* **Python 3.x**
* **Scikit-learn**
* **DVC**
* **MLflow**
* **DagsHub**
* **Pandas / NumPy**

---

## 📂 Project Structure

```
ml-pipeline-dvc-mlflow/
│
├── 📂 data
│   ├── raw/              # Original dataset (DVC tracked, not committed)
│   ├── processed/        # Preprocessed dataset (DVC output)
│
├── 📂 models             # Trained models (DVC tracked, not committed)
│
├── 📂 src
│   ├── preprocess.py     # Preprocessing script
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│
├── 📂 notebooks
│   ├── exploration.ipynb # Optional dataset exploration
│
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # Parameters for pipeline stages
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## 🔎 Pipeline Stages

### 🔹 Preprocessing

* Input: `data/raw/data.csv`
* Output: `data/processed/data.csv`
* Task: Column renaming and cleanup

```bash
dvc repro preprocess
```

### 🔹 Training

* Model: **Random Forest Classifier**
* Output: `models/model.pkl`
* Logs: Parameters + accuracy stored in **MLflow**

```bash
dvc repro train
```

### 🔹 Evaluation

* Loads trained model
* Evaluates accuracy on dataset
* Logs metrics in **MLflow**

```bash
dvc repro evaluate
```

---

## 🧠 Model Details

* **Algorithm:** Random Forest Classifier
* **Hyperparameters tracked:**

  * `n_estimators`
  * `max_depth`
  * `random_state`
* **Metric:** Accuracy

The trained model is stored as `models/model.pkl` (tracked by DVC).

---

## 🔬 Experiment Tracking with MLflow

Tracked during training & evaluation:

* **Parameters:** Random Forest hyperparameters
* **Metrics:** Accuracy
* **Artifacts:** Trained model (`.pkl`), processed datasets

👉 To view experiment history locally:

```bash
mlflow ui
```

Then open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 🔗 Live Project on DagsHub

You can explore the full pipeline runs, metrics, and data/model versions on **DagsHub**:
👉 [View the Live Project on DagsHub](https://dagshub.com/akshaygumma/ml-pipeline)

---

## 📌 Getting Started

### 1️⃣ Clone the repo

```bash
git clone https://github.com/your-username/diabetes-mlops-pipeline.git
cd diabetes-mlops-pipeline
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Pull data & models from DVC remote

```bash
dvc pull
```

### 4️⃣ Run the pipeline

```bash
dvc repro
```

### 5️⃣ Launch MLflow UI (to view experiments)

```bash
mlflow ui
```

Navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🖥️ Demo Flow

1. **Raw dataset** → stored in `data/raw` (tracked by DVC)
2. **Preprocessing stage** → generates `data/processed`
3. **Training stage** → trains Random Forest, saves `models/model.pkl`, logs params/metrics to MLflow
4. **Evaluation stage** → computes accuracy, logs to MLflow
5. **MLflow UI** → explore metrics, compare runs, check artifacts
6. **DagsHub link** → view pipeline, models, experiments online

---

## 🌟 What This Project Demonstrates

* ✔️ Ability to design **end-to-end ML pipelines**
* ✔️ Hands-on with **MLOps tools (DVC + MLflow + DagsHub)**
* ✔️ Strong focus on **reproducibility, versioning, and experiment management**
* ✔️ Practical ML project structuring, making it **production-ready**
* ✔️ Clear communication of work (good for collaboration & recruiters)

---

👨‍💻 Author: **Akshay Gumma**
🔗 Connect with me on [LinkedIn](https://www.linkedin.com/) | [DagsHub](https://dagshub.com/akshaygumma/ml-pipeline)
