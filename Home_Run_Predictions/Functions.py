
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def add_player_specific_features(data):
    # Calculate rolling averages for HR and AB over the past few seasons
    data['Rolling_HR'] = data.groupby('Name')['HR'].transform(lambda x: x.shift(1).rolling(window=3).mean())
    data['Rolling_AB'] = data.groupby('Name')['AB'].transform(lambda x: x.shift(1).rolling(window=3).mean())
    
    return data

def predict_home_runs(data):
    # Add player-specific features
    data = add_player_specific_features(data)

    # Replace NaN values with 0
    data = data.fillna(0)

    # Prepare features and target variable
    features = [
        'Age', 'AB', 'SLG', 'FB%',
        'HR/FB', 'K%', 'wRC+', 'Contact%', 'ISO',
        'HardHit%', 'Barrel%', 'LA', 'EV',
        'Rolling_HR',  # New feature
    ]

    target = 'HR'  # Home runs as the target variable

    # Define X and y
    X = data[features]
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models to try
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=1.0, random_state=42),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42)
    }

    # Fit and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'model': model, 'mse': mse, 'r2': r2}
        
        print(f"Model: {name}")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared Score: {r2}")
        print()

    # Select the best model after evaluating all
    best_model = None
    best_mse = float('inf')
    for name, result in results.items():
        if name == 'ElasticNet': #change this 
            best_model = result['model']
            best_mse = result['mse']
            print(f"Selected {name} as the best model.")
            break
        elif result['mse'] < best_mse:
            best_mse = result['mse']
            best_model = result['model']

    if best_model is None:
        print("ElasticNet not found. Selecting model with lowest MSE.")
        best_model = min(results.items(), key=lambda x: x[1]['mse'])[1]['model']

    print(f"Best Model: {type(best_model).__name__}")
    print(f"Best MSE: {best_mse}")

    # Use the best model for predictions
    next_season_predictions = best_model.predict(X)  # Predict using all available data
    data['predicted_HR'] = next_season_predictions  # Add predictions to DataFrame

    # Extract relevant information for predictions
    predictions_df = data[['Name', 'Season', 'predicted_HR']]  # Include Name and Season in output

    # Prepare leaderboard for next season predictions (2025)
    last_season_data = data[data['Season'] == 2024]  # Filter for players who played in 2024
    
    # Merge predictions with last season's player information based on Name
    leaderboard = last_season_data[['Name', 'Team', 'HR']].copy()  # Include actual HR from 2024
    
    leaderboard['Projected_HR'] = data.loc[data['Season'] == 2024, 'predicted_HR'].values  # Add predicted HR for 2025

    leaderboard['Projected_HR'] = np.round(leaderboard['Projected_HR'])
    
    leaderboard['Difference'] = leaderboard['Projected_HR'] - leaderboard['HR']  # Calculate difference


    leaderboard = leaderboard.sort_values("Projected_HR", ascending=False).reset_index(drop=True)
    
    print("\nPredicted Home Run Leaderboard for Next Season (2025):")
    print(leaderboard.head(30))  # Display top 30 projections

    leaderboard.to_csv('Home_Run_Leader_Board.csv')