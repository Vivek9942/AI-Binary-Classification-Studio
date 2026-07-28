# 🤖 AI Binary Classification Studio

An end-to-end, no-code Streamlit app to **upload, clean, train, evaluate, and predict** on any binary classification dataset — no ML experience required.

🔗 **Live App:** [[ai-binary-classification-studio.streamlit.app](ai-binary-classification-studio-ppqipctkuojykpozghsbaz
.streamlit.app)
](https://ai-binary-classification-studio-ppqipctkuojykpozghsbaz.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-005571?style=for-the-badge)
![License](https://img.shields.io/badge/License-Open%20Source-brightgreen?style=for-the-badge)

---

## ✨ Features

- 📂 **Upload any CSV** — the last column is auto-detected as the target label
- 🎯 **Auto binary-target validation** — rejects non-binary targets with a clear error
- 📄 **Dataset preview** — head/tail/random rows, dtypes, missing values, stats summary
- 🧹 **Automated preprocessing**
  - Removes duplicates and constant columns
  - Drops ID-like columns automatically
  - Converts booleans to integers
  - Fills missing values (median for numeric, mode for categorical)
  - Label-encodes categorical features and target
- ⚖️ **Automatic imbalance handling** with **SMOTE** when a class exceeds 70%
- 🤖 **Model training** — choose from:
  - Logistic Regression (with optimal threshold tuning via Precision-Recall curve)
  - Random Forest
  - Gradient Boosting
  - Extra Trees
- 📊 **Evaluation metrics** — Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- 🔮 **Live prediction** — interactive form generated from your dataset's features, with confidence score and decision threshold

---

## 🗂️ Repository Structure

```
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── loanapproval.csv           # Sample dataset
├── Binaryclassification.ipynb # Notebook (exploration/reference)
├── cleaned_data/               # Sample cleaned datasets
├── saved_models/               # Saved model artifacts
├── utils/                      # Helper utilities
└── README.md
```

---

## 🚀 Getting Started

### Run locally

```bash
git clone https://github.com/Vivek9942/AI-Binary-Classification-Studio.git
cd AI-Binary-Classification-Studio
pip install -r requirements.txt
streamlit run app.py
```

### Run on Streamlit Cloud

Already deployed → just visit the [live app link](https://ai-binary-classification-studio-d73kncushjngirdaj5top7.streamlit.app).

---

## 📋 How to Use

1. **Upload Dataset** — upload a CSV where the **last column** is your binary target (e.g. `Yes`/`No`,`Pass`/`Fail` etc.).
2. **Dataset Preview** — inspect rows, data types, and summary statistics.
3. **Data Preprocessing** — automatically cleaned, encoded, and ready for training. Download the cleaned CSV if needed.
4. **Model Training** — pick a model and click **Train**. View class balance, SMOTE application (if needed), and performance metrics.
5. **Prediction** — fill in the generated form with new values and get an instant prediction with confidence score.

---

## 🛠️ Tech Stack

| Purpose | Library |
|---|---|
| Web app / UI | [Streamlit](https://streamlit.io/) |
| Data handling | [pandas](https://pandas.pydata.org/) |
| ML models & metrics | [scikit-learn](https://scikit-learn.org/) |
| Imbalanced data handling | [imbalanced-learn](https://imbalanced-learn.org/) (SMOTE) |

---

## 📌 Notes

- Only supports **binary classification** — datasets with a target that isn't exactly 2 classes will be rejected.
- Columns with "id" in their name are auto-dropped as they don't carry predictive signal.
- Known leakage-prone columns (e.g. `Score`, `Marks`, `Final_Exam_Score`) are auto-excluded from training features.

---

## 👤 Author

**Vivek Pandey** ([@Vivek9942](https://github.com/Vivek9942))

---

## 📄 License

This project is open source and available for personal/educational use.
