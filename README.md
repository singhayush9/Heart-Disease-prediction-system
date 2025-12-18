# Heart Disease Prediction System

An end-to-end Machine Learning project that predicts the likelihood of heart disease based on clinical parameters.  
This repository includes:

- A trained ML model
- Preprocessing pipeline
- Clearly defined medical features
- A Python/Streamlit based user interface
- Reusable utilities for loading, predicting, and extending the system

---

## Project Overview

Cardiovascular disease is one of the leading causes of death worldwide.  
Early risk detection using data-driven methods can help doctors and patients make better decisions.

This project builds a **Heart Disease Prediction System** that:

- Takes patient information as input (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- Applies preprocessing and feature engineering
- Uses a trained machine learning model to predict **whether the person is at risk of heart disease**
- Provides a simple interface to interact with the model

---

## Features

-  **Trained ML model** stored as `heart_disease_model.joblib`
-  **Well-defined features** with descriptions and valid ranges in `src/data/feature_definitions.py`
-  **Preprocessing pipeline** for cleaning and transforming input data
-  **User Interface layer** (CLI / app) for easy usage
-  Modular, clean code structure (`src/models`, `src/preprocessing`, `src/ui`, `src/utils`)
-  Easy to retrain with new data

---

## Machine Learning Model

The trained model (stored in `models/heart_disease_model.joblib`) is built using traditional ML algorithms such as:

- Logistic Regression / Random Forest / Gradient Boosting (depending on configuration)

The system:

1. Loads the model
2. Applies the same preprocessing steps used during training
3. Generates a prediction (e.g. **0 = Low Risk**, **1 = High Risk**)

---

## Project Structure


Heart-Disease-prediction-system/
│
├── models/
│   └── heart_disease_model.joblib      # Saved trained model
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── feature_definitions.py      # Feature metadata: description, type, valid ranges
│   │
│   ├── models/                         # Model training / loading code
│   │   └── ... 
│   │
│   ├── preprocessing/                  # Preprocessing & feature engineering
│   │   └── ...
│   │
│   ├── ui/                             # UI related code (forms / helpers)
│   │   └── ...
│   │
│   ├── utils/                          # Utility functions (logging, config, etc.)
│   │   ├── __init__.py
│   │   └── ...
│   │
│   └── __init__.py
│
├── app.py                              # App entry point (e.g., Streamlit / Web UI)
├── main.py                             # Main script (CLI / pipeline / demo)
├── requirements.txt                    # Python dependencies
└── README.md

# All feature metadata is defined in:
All feature metadata is defined in:

Each feature includes:

description – human-readable explanation

type – numeric / categorical

valid_range / allowed_values – expected value range

Typical features may include:

Age

Sex

Chest pain type

Resting blood pressure

Cholesterol level

Fasting blood sugar

Resting ECG results

Maximum heart rate achieved

Exercise induced angina

ST depression

…and more depending on configuration.

## How To Run This Project
Clone the Repository

Open a terminal / PowerShell and run:

git clone https://github.com/singhayush9/Heart-Disease-prediction-system.git

cd Heart-Disease-prediction-system

pip install -r requirements.txt

## Run the Application

Depending on how you want to interact with the model, you can:

 Option A: Run the main Python script

python main.py

 Option B: Run the app (e.g., Streamlit / Web UI)

If app.py uses Streamlit or a similar framework:# If Streamlit-based:

streamlit run app.py

Or if it's a plain Python UI script:

python app.py

## Typical Workflow

User provides clinical inputs using the UI or CLI

Inputs are validated using feature_definitions.py

Preprocessing transforms raw inputs into model-readable format

The trained ML model generates a prediction

Result is displayed as:

Probability of heart disease

Risk label (e.g., Low, Moderate, High)




