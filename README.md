# Diabetic Readmission Risk Prediction

## Goal of the Project

The goal of this project was to build an end-to-end machine learning workflow that predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days.

Hospital readmissions are an important issue for healthcare systems because they increase costs and often indicate gaps in care planning. I wanted to explore whether patient encounter data could be used to estimate readmission risk and what factors might influence it.

The project covers the full pipeline starting from data exploration all the way to an interactive prediction app.

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
7. Build a simple Streamlit application to demonstrate predictions

The goal was not just to train a model but to build something closer to a **complete ML workflow**.

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

It loads the dataset, applies the preprocessing functions, splits the data into training and testing sets, and trains the models. After evaluation, the best performing model is saved so it can later be used by the Streamlit app.

---

### `streamlit_app.py`

This is a small interactive interface built with Streamlit.

It allows someone to enter a hypothetical patient profile such as age group, number of procedures, lab tests, and hospital visits. The app then uses the trained model to estimate the probability of readmission and displays the predicted risk.

I mainly added this so the project isn't just a notebook experiment but something you can actually interact with.

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

Overall, the project shows that while predicting readmissions is challenging due to data imbalance and missing contextual factors, machine learning models can still provide useful signals about patient risk patterns.

---

## Running the Project

Install the required dependencies and train the model first.

-python main.py
Then launch the Streamlit interface:
-streamlit run app/streamlit_app.py

This will open a local web app where you can experiment with different patient profiles and see the predicted readmission risk.

---

## Final Thoughts

This project was mainly an exercise in building a **complete machine learning workflow** rather than focusing only on model accuracy.

It includes data exploration, preprocessing pipelines, model comparison, explainability, evaluation improvements, and a simple application layer to demonstrate predictions.

There are definitely many ways the project could be extended, but it was a good way for me to practice putting all the pieces of an ML project together.
