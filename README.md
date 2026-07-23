# 🤖 AI Binary Classification Studio

An interactive **Streamlit** web app that lets you upload *any* binary classification dataset and run a complete, end-to-end machine learning workflow — from raw CSV to live predictions — without writing a single line of code.

**🔗 Live App:** [ai-binary-classification-studio-gdljuxdv5nggjhwtwu2hmg.streamlit.app](https://ai-binary-classification-studio-gdljuxdv5nggjhwtwu2hmg.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Overview

Upload a CSV file, pick your target column, and the app automatically cleans, encodes, trains, evaluates, and serves predictions from the best-performing model — all through a clean, tab-based UI.

## 🚀 Features

### 📂 Upload Dataset
- Upload any CSV file with a binary (two-class) target
- Instant summary metrics: rows, columns, missing values, duplicate rows
- Auto-detects candidate binary target columns
- Custom class labels for numeric (0/1) targets

### 📄 Dataset Preview
- View first / last / random 5 rows
- Column-level info: data type, missing values, unique values
- Full statistical summary (`describe()`)

### 🧹 Data Preprocessing
Automatic, no-config cleaning pipeline:
- Removes duplicate rows
- Drops constant (single-value) columns
- Drops likely ID columns
- Converts boolean columns to integers
- Fills missing numeric values with the median
- Fills missing categorical values with the mode
- Label-encodes categorical features and the target column
- Download the cleaned dataset as CSV

### 🤖 Model Training
Trains and compares three classifiers side by side:
- Logistic Regression (with feature scaling)
- Random Forest Classifier
- Gradient Boosting Classifier

Automatically:
- Splits data into train/test sets (stratified when possible)
- Guards against target leakage from obvious score/marks columns
- Reports Accuracy, Precision, Recall, and F1 Score for each model
- Selects and highlights the best-performing model

### 🔮 Prediction
- Dynamically generated input form based on your dataset's features
- Handles both categorical (dropdown) and numerical (numeric input) fields
- Predicts the outcome using the best trained model
- Displays prediction confidence with a progress bar and percentage

---

## 🛠️ Tech Stack

| Category            | Tools                                   |
|---------------------|------------------------------------------|
| Language            | Python                                   |
| Web Framework       | [Streamlit](https://streamlit.io/)       |
| Data Handling       | pandas                                   |
| Machine Learning    | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| Preprocessing       | LabelEncoder, StandardScaler             |

## 📁 Project Structure

```
AI-Binary-Classification-Studio/
├── app.py                     # Main Streamlit application
├── Binaryclassification.ipynb # Notebook version / experimentation
├── loanapproval.csv           # Sample dataset
├── requirements.txt           # Python dependencies
├── cleaned_data/              # Saved cleaned datasets
├── saved_models/               # Saved trained models
├── utils/                     # Helper utilities
└── README.md
```

## ⚙️ Installation & Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vivek9942/AI-Binary-Classification-Studio.git
   cd AI-Binary-Classification-Studio
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # On Windows
   source venv/bin/activate   # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open your browser at `http://localhost:8501`

## 📖 How to Use

1. Go to **📂 Upload Dataset** and upload a CSV with a binary target column.
2. Select the target column and (if numeric) label its two classes.
3. Explore your data in **📄 Dataset Preview**.
4. Let the app auto-clean your data in **🧹 Data Preprocessing** and download the cleaned CSV if needed.
5. Head to **🤖 Model Training** to train and compare Logistic Regression, Random Forest, and Gradient Boosting models.
6. Use **🔮 Prediction** to enter new feature values and get a live prediction with confidence score.

## 📊 Sample Dataset

A sample dataset, `loanapproval.csv`, is included so you can try the app immediately without needing your own data.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Vivek9942/AI-Binary-Classification-Studio/issues) or open a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License.

## 👤 Author

**Vivek Pandey**
GitHub: [@Vivek9942](https://github.com/Vivek9942)

---

⭐ If you found this project useful, consider giving it a star on GitHub!
