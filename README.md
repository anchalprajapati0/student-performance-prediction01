# 🎓 Student Performance Prediction System

## 📌 Project Overview

This project predicts a student's final academic performance (G3) using Machine Learning.

The model is trained on the Student Performance Dataset and deployed using Streamlit.

---

## 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit
- Joblib

---

## 📂 Project Structure

```
student_performance_prediction/
│
├── Dataset/
│   └── student-por[1].csv
│
├── Model/
│   ├── student_model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
│
├── Streamlit_App/
│   └── app.py
│
├── train_model.py
├── requirements.txt
└── README.md
```

---

## 📊 Machine Learning Workflow

- Load Dataset
- Data Preprocessing
- Label Encoding
- Feature Scaling
- Train-Test Split
- XGBoost Model Training
- Model Evaluation
- Save Model
- Streamlit Deployment

---

## 📈 Model Performance

| Metric | Value |
|--------|--------|
| R² Score | 0.8268 |
| MAE | 0.7706 |
| RMSE | 1.2997 |

---

## 🎯 Features

✔ Student Information

✔ Family Information

✔ Study Details

✔ Lifestyle Information

✔ Previous Academic Performance

✔ Final Grade Prediction

---

## ▶️ Run the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
cd Streamlit_App
python -m streamlit run app.py
```

---

## 👩‍💻 Developer

**Anchal Prajapati**

B.Tech CSE

United College of Engineering & Research
