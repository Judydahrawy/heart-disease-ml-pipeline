# Heart Disease Machine Learning Pipeline ❤️🩺

An end-to-end machine learning pipeline for analyzing, predicting, and visualizing heart disease risks using the UCI Heart Disease Dataset. This project features comprehensive exploratory data analysis, dimensionality reduction, clustering, and classification models, all wrapped up in an interactive Streamlit web application.

## 🚀 Features

- **Data Preprocessing**: Handling missing values, one-hot encoding for categorical variables, and feature scaling.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) to retain essential variance while reducing features.
- **Feature Selection**: Recursive Feature Elimination (RFE) and Chi-Square tests to identify the most significant predictors.
- **Supervised Learning**: Logistic Regression, Decision Trees, Random Forest, and Support Vector Machines (SVM) for classification.
- **Unsupervised Learning**: K-Means and Hierarchical Clustering for pattern discovery.
- **Hyperparameter Tuning**: Model optimization using `GridSearchCV` and `RandomizedSearchCV`.
- **Interactive UI**: A Streamlit web application for real-time predictions and data visualization.

## 📁 Project Structure

```text
heart-disease-ml-pipeline/
│── data/
│   ├── heart_disease.csv            # The dataset (downloaded via script)
│── notebooks/
│   ├── heart_disease_pipeline.ipynb # Consolidated ML pipeline notebook
│── models/
│   ├── final_model.pkl              # Exported optimized model
│── scripts/
│   ├── download_data.py             # Script to fetch UCI dataset
│── ui/
│   ├── app.py                       # Streamlit application source code
│── results/
│   ├── evaluation_metrics.txt       # Model performance scores
│── requirements.txt                 # Project dependencies
│── README.md                        # Project documentation
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Judydahrawy/heart-disease-ml-pipeline.git
   cd heart-disease-ml-pipeline
   ```

2. **Install the dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Data:**
   Run the helper script to fetch the dataset from the UCI Machine Learning Repository:
   ```bash
   python scripts/download_data.py
   ```

## 🏃‍♂️ Running the Project

### Exploring the Notebook
To view the machine learning pipeline, data cleaning, and model training steps, launch Jupyter Notebook:
```bash
jupyter notebook notebooks/heart_disease_pipeline.ipynb
```

### Running the Web App
To start the Streamlit interactive UI and make real-time predictions:
```bash
streamlit run ui/app.py
```

## 📊 Dataset
The project utilizes the [Heart Disease UCI Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). It contains multiple clinical attributes (e.g., age, cholesterol levels, max heart rate) used to predict the presence of heart disease in patients.
