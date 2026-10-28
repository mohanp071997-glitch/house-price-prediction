# House Price Prediction

## Project Overview

This project predicts house prices based on property features using Machine Learning and provides an interactive web interface built with Streamlit.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Linear Regression
- Streamlit
- Joblib

## Features

- Exploratory Data Analysis (EDA)
- Missing-value handling
- Categorical feature encoding
- Machine Learning model training
- Model evaluation
- Saved model for reuse
- Interactive house price prediction web application

## Model Performance

- MAE: approximately 5.12 Lakhs
- RMSE: approximately 6.40 Lakhs
- R² Score: approximately 0.914

## How to Run

1. Install the required libraries:

   pip install -r requirements.txt

2. Run the Streamlit application:

   streamlit run app.py

## Project Files

- `main.py` — data analysis and model training
- `predict.py` — terminal-based prediction
- `app.py` — Streamlit web application
- `house_prices_1000_records.csv` — practice dataset
- `house_price_model.pkl` — saved model
- `requirements.txt` — required libraries

## Important Note

The dataset contains synthetic practice data. Predictions are for learning purposes and should not be treated as actual real-estate valuations.
