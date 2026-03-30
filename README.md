# 🏠 Housing Price Prediction App

A professional, end-to-end machine learning application that predicts California residential property values using a Random Forest model and a modern, high-end web dashboard.

![Project Preview](https://img.shields.io/badge/ML-Random%20Forest-blueviolet?style=for-the-badge)
![UI](https://img.shields.io/badge/UI-Glassmorphism-blue?style=for-the-badge)
![Backend](https://img.shields.io/badge/Backend-Flask-lightgrey?style=for-the-badge)

## ✨ Features

- **Modern Dashboard**: Premium dark-mode interface with glassmorphism effects.
- **Real-time Inference**: Input property details and get instant price estimates.
- **Advanced Preprocessing**: Comprehensive feature engineering including log transformations and geographic feature creation.
- **Robust Model**: Powered by a Scikit-Learn `RandomForestRegressor` for high-accuracy predictions.

## 📁 Repository Structure

- `app.py`: The Flask server and prediction API.
- `train_model.py`: Training pipeline that exports the optimized model and scaler.
- `index.py`: Exploratory Data Analysis (EDA) and model experimentation script.
- `templates/`: HTML structures for the frontend.
- `static/`: CSS styling and JavaScript interactivity.
- `housing.csv`: The California Housing Prices dataset.
- `requirements.txt`: Project dependencies.

## 🚀 Setup and Installation

### 1. Clone & Navigate
```bash
git clone https://github.com/yourusername/Housing-Price-Prediction-Model.git
cd Housing-Price-Prediction-Model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install flask joblib  # Extra dependencies for the UI
```

### 3. Train the Model
Before running the UI, you must train the model to generate the required `.joblib` files:
```bash
python3 train_model.py
```

### 4. Start the Application
Launch the Flask web server:
```bash
python3 app.py
```
Open your browser to `http://127.0.0.1:5000` to interact with the dashboard.

## 📊 Data Insights

The model utilizes the [California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices), analyzing features such as:
- **Location**: Longitude, Latitude, and Ocean Proximity.
- **Property Specs**: Total Rooms, Bedrooms, and Housing Age.
- **Demographics**: Population, Households, and Median Income.

## 🛠️ Technology Stack
- **Languages**: Python, HTML, JavaScript
- **ML Frameworks**: Scikit-Learn, Pandas, NumPy
- **Web Backend**: Flask
- **Styling**: Vanilla CSS (Modern CSS3 with Glassmorphism)
