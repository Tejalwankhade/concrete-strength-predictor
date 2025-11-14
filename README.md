🧱 Revisiting a Concrete Strength Regression – Streamlit App

This project builds a machine learning regression model to predict Concrete Compressive Strength (csMPa) using eight key ingredients and curing age.
A trained model (Concrete_Strength_Model.pkl) is integrated into a Streamlit web application to allow interactive predictions.

📌 Project Structure
│── streamlit_app.py
│── Concrete_Strength_Model.pkl
│── requirements.txt
│── README.md

🧪 Dataset Overview

Dataset name: Revisiting a Concrete Strength Regression
Target variable:

csMPa – Concrete compressive strength (in MPa)

Input features:

cement

slag

flyash

water

superplasticizer

coarseaggregate

fineaggregate

age

🚀 Streamlit App Features

✔️ Loads trained model (Concrete_Strength_Model.pkl)

✔️ Takes 8 input features from user

✔️ Predicts concrete strength (csMPa)

✔️ Shows model type and pipeline steps

✔️ Supports models trained using XGBoost, Scikit-Learn, Joblib, or Pickle

