# 🏦 Bank Customer Churn Prediction

## 📌 Project Overview

This project predicts whether a bank customer is likely to leave the bank (churn) or continue using the services.

It uses machine learning to analyze customer details like credit score, balance, activity status, and more to make predictions.

The project also includes an interactive web application built using Streamlit.

---

## 🚀 Features

* Predict customer churn using trained ML model
* Interactive and clean UI using Streamlit
* Real-time prediction with probability score
* User-friendly input system (sliders, dropdowns)
* Displays churn vs stay probability

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Matplotlib, Seaborn

Dependencies used:
(see requirements.txt) 

---

## 📊 Dataset

* Bank Customer Churn Dataset
* Contains customer details such as:

  * Credit Score
  * Age
  * Geography
  * Balance
  * Number of Products
  * Active Status
  * Estimated Salary

---

## ⚙️ How It Works

1. User enters customer details in the Streamlit app
2. Data is encoded and processed
3. Trained model (`churn_model.pkl`) is used
4. Prediction is generated
5. Result + probability is displayed

---

## 💻 How to Run the Project

### 1️⃣ Clone Repository

```bash id="h4j3k1"
git clone https://github.com/your-username/bank-churn-prediction.git
cd bank-churn-prediction
```

### 2️⃣ Install Dependencies

```bash id="k92nsl"
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash id="p82kdn"
streamlit run app.py
```

---

## 📈 Model Workflow

* Data Cleaning
* Feature Encoding
* Model Training
* Model Evaluation
* Model Deployment

---

## 📌 Output Example

* Shows whether customer will churn or stay
* Displays probability percentage
* Easy to understand visual output

---

## 💡 Future Improvements

* Add more features for better accuracy
* Deploy on cloud platforms
* Improve UI with more visual insights

---

## ⭐ Note

This project is created for learning and portfolio purposes.
