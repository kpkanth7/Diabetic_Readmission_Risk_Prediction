# Diabetic Readmission Risk Prediction

**🚀 Live Application:** [https://diabetic-predict-readmission.onrender.com](https://diabetic-predict-readmission.onrender.com)

## Goal of the Project

The goal of this project was to build an end-to-end machine learning workflow that predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days.

Hospital readmissions are an important issue for healthcare systems because they increase costs and often indicate gaps in care planning. I wanted to explore whether patient encounter data could be used to estimate readmission risk and what factors might influence it.

> **⚠️ Disclaimer: Strictly Educational**  
> *This project is strictly for educational purposes and is a personal portfolio piece. It should not be used for actual clinical decision-making. The data is limited, and the model accuracies are not perfect. While there is some diversity in the dataset, there is not enough diversity and volume of data to fully pick up all the complex real-world patterns required to help models learn robustly.*

The project covers the full pipeline starting from data exploration all the way to an interactive, blazing-fast web application.

---

## Dataset

This project uses the **UCI Diabetes 130-US Hospitals dataset (1999-2008)**.

It contains information about more than **100,000 hospital encounters** of diabetic patients and includes:

- patient demographics  
- hospital stay information  
- number of procedures and lab tests  
- medication usage  
- prior hospital visits  
- diagnosis counts  

The target variable is whether the patient was **readmitted within 30 days**.

---

## How the Project Works

The workflow roughly follows these steps:

1. Explore and understand the dataset
2. Clean and preprocess the features
3. Train several machine learning models
4. Compare model performance
5. Interpret model behavior using SHAP
6. Tune the prediction threshold for the imbalanced dataset
7. Build a custom, high-performance web application (FastAPI + Vanilla JS/HTML/CSS) to serve predictions

The aim was not just to train a model but to build something closer to a **complete ML workflow and production-ready application**.

---

## Main Components

### `src/preprocess.py`

This file contains the data preprocessing logic used throughout the project.

It handles things like:

- cleaning raw dataset values  
- formatting categorical variables  
- preparing numeric columns  
- making sure the training pipeline and the prediction pipeline use the exact same transformations

Keeping preprocessing in one place made it easier to reuse the same logic both during training and when making predictions in the app.

---

### `main.py`

This script trains the machine learning model.

It loads the dataset, applies the preprocessing functions, splits the data into training and testing sets, and trains the models. After evaluation, the best performing model is saved so it can later be used by the web app.

---

### `app/server.py` & `static/` (The Web Application)

Instead of a basic prototyping framework, I built a custom, highly-performant web application to serve the model:

- **Backend (`app/server.py`)**: A lightning-fast **FastAPI** server that loads the pickled model and exposes a `/predict` API endpoint.
- **Frontend (`static/`)**: A bespoke user interface built with pure **HTML, CSS, and Javascript**. It features modern glassmorphism design, dynamic animations, and instantaneous DOM updates without page reloads.

This architecture provides a much smoother, dynamic, and realistic experience for interacting with the machine learning model.

---

### Notebooks

The notebooks document different stages of the project.

**eda.ipynb**
Explores the dataset and looks at distributions, missing values, and some basic relationships between variables.

**model_comparison.ipynb**
Trains and compares multiple models including logistic regression, random forest, and gradient boosting.

**explainability.ipynb**
Uses SHAP to understand which features influence the model predictions the most.

**model_improvement.ipynb**
Looks at evaluation metrics in more detail and explores threshold tuning because the dataset is somewhat imbalanced.

---

## Results

The dataset contains **101,766 patient encounters** with **47 variables** describing patient demographics, hospital visits, procedures, medications, and diagnoses.

I trained and compared a few models including **Logistic Regression, Random Forest, and Gradient Boosting**.

Some observations from the experiments:

- **Logistic Regression** achieved the highest recall, meaning it was better at identifying potential readmissions, but its overall predictive strength was limited.
- **Random Forest** produced high accuracy but struggled to detect the positive readmission cases because of the class imbalance.
- **Gradient Boosting** achieved the **best ROC-AUC score (~0.68)** and overall provided the most balanced performance among the tested models.

Since the dataset is highly imbalanced, accuracy alone was not a reliable metric. Instead, I focused more on **ROC-AUC, recall, and precision-recall behavior** when comparing models.

Using SHAP explainability, the model indicated that features such as:

- number of inpatient visits  
- number of emergency visits  
- number of diagnoses  
- number of medications  
- length of hospital stay  

tended to have stronger influence on predicting readmission risk.

Because positive readmissions are relatively rare, I also explored **threshold tuning** to better balance precision and recall, which helped improve the model's ability to flag potential readmissions.

Overall, the project shows that while predicting readmissions is challenging due to data limitations, machine learning models can still provide useful signals about patient risk patterns.

---

## Running the Project Locally

Install the required dependencies and train the model first.

```bash
pip install -r requirements.txt
python main.py
```

Then launch the FastAPI web server:

```bash
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

This will open a local web app at `http://localhost:8000` where you can interact with the beautiful UI, experiment with different patient profiles, and see the predicted readmission risk instantly.

---

## Final Thoughts

This project was mainly an exercise in building a **complete machine learning workflow** rather than focusing only on model accuracy.

It includes data exploration, preprocessing pipelines, model comparison, explainability, evaluation improvements, and a modern web application architecture to demonstrate predictions.

There are definitely many ways the project could be extended, but it was a great way to practice putting all the pieces of an ML project together from data ingestion to a live, styled production deployment.
