# Shopping Intention Prediction App

This is a Streamlit web app that predicts whether an online shopper will complete a purchase based on session behavior and technical features. It uses a pre-trained Random Forest machine learning model.

---

## Features

- Predicts user purchase intent based on:
  - Pages visited
  - Time spent on product, info, and admin pages
  - Bounce and exit rates
  - Device/browser info
  - Special days, weekends, etc.
- Clean, interactive UI using Streamlit
- Human-readable inputs (dropdowns, sliders, and friendly labels)

---

## Files in this Repository

| File | Description |
|------|-------------|
| `app.py` | Streamlit app source code |
| `rf_model.pkl` | Pre-trained Random Forest model |
| `scaler.pkl` | Fitted scaler for input normalization |
| `online_shoppers_intention.csv` | Dataset used for model training |
| `requirements.txt` | All required Python packages |
| `README.md` | This file |

---

## Run the App Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/smriti0210/Shopping-Intension-App.git
   cd Shopping-Intension-App

   Create a virtual environment:

## Create a virtual environment:
python -m venv shopper-env
.\shopper-env\Scripts\Activate.ps1   # For Windows

## Install dependencies:
pip install -r requirements.txt

## Launch the app:
streamlit run app.py

## Dataset Source
Online Shoppers Purchasing Intention Dataset

## Source:
UCI Machine Learning Repository

## Model Info
Algorithm: Random Forest
Preprocessing: StandardScaler
Accuracy: 90.02%

## Author
Made with ❤️ by Smriti Jha
