# Housing Price Prediction Model

This repository contains a machine learning model for predicting housing prices. The project was created as a learning exercise to understand the fundamentals of data preprocessing, feature engineering, model selection, and evaluation.

## Project Overview

The objective of this project is to build a predictive model that estimates the prices of houses based on various features. The model is trained on a dataset containing information about different houses, such as their size, location, number of rooms, and other relevant attributes.

## Data

The dataset used for this project is the [California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices). It includes the following features:

- **longitude:** The longitude coordinate of the house.
- **latitude:** The latitude coordinate of the house.
- **housing_median_age:** The median age of the houses in the area.
- **total_rooms:** The total number of rooms in the house.
- **total_bedrooms:** The total number of bedrooms in the house.
- **population:** The population of the area.
- **households:** The number of households in the area.
- **median_income:** The median income of households in the area.
- **median_house_value:** The median value of the houses (target variable).
- **ocean_proximity:** The proximity of the house to the ocean.

## Project Structure

The project is organized as follows:
- `housing.csv`: The dataset containing housing information.
- `index.py`: The main script for data preprocessing, model training, and evaluation.
- `requirements.txt`: File listing the required Python packages.
- `README.md`: Project documentation.

## Setup and Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Housing_Price_Prediction_Model.git
    cd Housing_Price_Prediction_Model
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the model, execute the following command:
```bash
python index.py


