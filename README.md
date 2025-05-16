# employee_attrition
Employee Attrition Analysis and Prediction

# Employee Attrition and Performance Prediction

This project focuses on building machine learning models to predict two key HR metrics:
1. Whether an employee is likely to **leave the company** (Attrition Prediction).
2. The **performance rating** of an employee (Performance Prediction).

Both models are deployed using a **Streamlit-based web application** that allows interactive predictions.

---

## 📁 Project Structure

```
├── attrition_model.ipynb                  # Preprocessing + training XGBoost model for attrition
├── model_employee_attrition_.ipynb        # Additional modeling work for attrition
├── employee_attrition_eda.ipynb           # Exploratory Data Analysis for attrition dataset
├── performance_rating_model.ipynb         # Preprocessing + training XGBoost model for performance rating
├── performance_rating_eda.ipynb           # EDA for employee performance dataset
├── attrition_xgboost_model.pkl            # Saved XGBoost model for attrition prediction
├── performance_rating_xgboost_model.pkl   # Saved XGBoost model for performance prediction
├── employee_attrition_performance_prediction_app.py  # Streamlit app for user interaction
```

---

## 🔍 Features Used

### Attrition Prediction
Features selected for the attrition model:
- Overtime
- Marital Status (Single)
- Years in Current Role
- Monthly Income
- Job Role (e.g., Sales Representative, Research Director)
- Years With Current Manager
- Stock Option Level
- Job Involvement
- Business Travel Frequency
- Job Satisfaction
- Environment Satisfaction
- Department (e.g., Research & Development)

### Performance Rating Prediction
Features selected for the performance model:
- Percent Salary Hike
- Years in Current Role
- Job Role (e.g., Manager)
- Relationship Satisfaction
- Environment Satisfaction
- Job Involvement
- Distance From Home
- Business Travel
- Education and Field
- Job Level
- Gender
- Overtime
- Work-Life Balance
- Training Times Last Year

---

## 🧠 Models

- **XGBoost Classifier** was used for both predictions due to its performance on structured data.
- Models were trained after performing **feature engineering and selection**, and saved using `joblib`.

---

## 🚀 How to Run

1. Install required packages:
    ```bash
    pip install streamlit joblib numpy
    ```

2. Make sure the following files are in the same directory:
    - `employee_attrition_performance_prediction_app.py`
    - `attrition_xgboost_model.pkl`
    - `performance_rating_xgboost_model.pkl`

3. Run the app:
    ```bash
    streamlit run employee_attrition_performance_prediction_app.py
    ```

4. Use the **two tabs**:
    - **Attrition** tab for predicting employee attrition.
    - **Performance** tab for predicting performance rating.

---

## ✅ Output

- For Attrition Prediction:
  - "Employee Will leave" or "Employee Will not leave"
  
- For Performance Rating:
  - "Employee Good Rating" or "Employee Average Rating"

---

## 📊 Notebooks Overview

- `*_eda.ipynb`: Exploratory Data Analysis including distribution plots, correlations, and insights.
- `*_model.ipynb`: Data preprocessing, training/testing, and model evaluation.

---

## 🛠️ Future Improvements

- Add feature scaling and categorical encoding steps dynamically based on new data.
- Enable batch prediction with CSV upload.
- Improve UI for easier usage with dropdowns/radio buttons for categorical features.
- Integrate model explanation using SHAP or LIME.

