import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

def Format_Data(DataType):
    DataType = DataType.copy()
    # Apply log transformations
    DataType['total_rooms'] = np.log(DataType['total_rooms'] + 1)
    DataType['total_bedrooms'] = np.log(DataType['total_bedrooms'] + 1)
    DataType['population'] = np.log(DataType['population'] + 1)
    DataType['households'] = np.log(DataType['households'] + 1)
    
    # One-hot encoding for ocean_proximity
    # We need to make sure we have all possible categories
    ocean_categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    for cat in ocean_categories:
        DataType[cat] = (DataType['ocean_proximity'] == cat).astype(int)
    
    DataType = DataType.drop(['ocean_proximity'], axis=1)
    
    DataType['bedroom_radio'] = DataType['total_bedrooms'] / DataType['total_rooms']
    DataType['household_rooms'] = DataType['total_rooms'] / DataType['households']
    
    return DataType

def train():
    data_path = "housing.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    data = pd.read_csv(data_path)
    data.dropna(inplace=True)
    
    X = data.drop(['median_house_value'], axis=1)
    Y = data['median_house_value']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    train_data = X_train.join(Y_train)
    train_data = Format_Data(train_data)
    
    X_train_final = train_data.drop('median_house_value', axis=1)
    Y_train_final = train_data['median_house_value']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    
    # Using a slightly faster training for demonstration if needed, 
    # but the user had a grid search. Let's do a decent RF model.
    forest = RandomForestRegressor(n_estimators=100, random_state=42)
    forest.fit(X_train_scaled, Y_train_final)
    
    # Save the model and scaler
    joblib.dump(forest, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Save column names to ensure consistency in inference
    joblib.dump(X_train_final.columns.tolist(), 'feature_columns.joblib')
    
    print("Model and scaler saved successfully!")
    
    # Evaluation
    test_data = X_test.join(Y_test)
    test_data = Format_Data(test_data)
    X_test_final = test_data.drop('median_house_value', axis=1)
    Y_test_final = test_data['median_house_value']
    X_test_scaled = scaler.transform(X_test_final)
    
    score = forest.score(X_test_scaled, Y_test_final)
    print(f"Test Score: {score}")

if __name__ == "__main__":
    train()
