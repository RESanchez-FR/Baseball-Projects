
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


class HomeRunPredictor:
    def __init__(self):
        # Initialize base features
        self.base_features = [
            'Age', 'AB', 'RBI', 'SLG', 'OBP', 'FB%',
            'HR/FB', 'LD%', 'K%', 'BB%', 'wOBA', 'wRC+',
            'O-Contact%', 'Z-Contact%', 'HardHit%', 
            'Barrel%', 'LA', 'EV'
        ]
        # Initialize extended features list
        self.features = self.base_features.copy()
        self.target = 'HR'
        self.simple_model = None
        self.complex_model = None
        self.best_params = None
        
    def add_player_specific_features(self, df):
        """Add rolling averages for player-specific features"""
        df = df.copy()
        # Calculate rolling averages
        df['Rolling_HR'] = df.groupby('Name')['HR'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        df['Rolling_AB'] = df.groupby('Name')['AB'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        # Update features list if not already included
        if 'Rolling_HR' not in self.features:
            self.features.extend(['Rolling_HR', 'Rolling_AB'])
        return df

    def prepare_data(self, df):
        """Prepare data for modeling"""
        # Add player-specific features
        df = self.add_player_specific_features(df)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Select features and target
        X = df[self.features]
        y = df[self.target]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_simple_model(self, X_train, y_train):
        """Train a simple linear regression model"""
        self.simple_model = LinearRegression()
        self.simple_model.fit(X_train, y_train)
        return self.simple_model

    def tune_complex_model(self, X_train, y_train):
        """Tune XGBoost model using GridSearchCV"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        model = XGBRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        self.complex_model = grid_search.best_estimator_
        return self.complex_model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            'mse': mse,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }

    def generate_predictions(self, df, model):
        """Generate predictions for next season"""
        # Add player-specific features and handle missing values
        df_prepared = self.add_player_specific_features(df)
        df_prepared = df_prepared.fillna(0)
        
        # Make predictions
        X = df_prepared[self.features]
        df_prepared['predicted_HR'] = np.round(model.predict(X)).astype(int)
        
        # Create leaderboard for next season
        last_season_data = df_prepared[df_prepared['Season'] == 2024]
        leaderboard = last_season_data[['Name', 'Team', 'HR']].copy()
        leaderboard['Projected_HR'] = df_prepared.loc[df_prepared['Season'] == 2024, 'predicted_HR'].values
        leaderboard['Difference'] = leaderboard['Projected_HR'] - leaderboard['HR']
        
        return leaderboard.sort_values("Projected_HR", ascending=False).reset_index(drop=True)

    def run_full_pipeline(self, data):
        """Run the complete prediction pipeline"""
        # Prepare data
        print("Preparing data...")
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        # Train and evaluate simple model
        print("\nTraining Simple Model...")
        simple_model = self.train_simple_model(X_train, y_train)
        simple_metrics = self.evaluate_model(simple_model, X_test, y_test)
        print("Simple Model Metrics:", simple_metrics)
        
        # Train and evaluate complex model
        print("\nTraining Complex Model with Grid Search...")
        complex_model = self.tune_complex_model(X_train, y_train)
        complex_metrics = self.evaluate_model(complex_model, X_test, y_test)
        print("Complex Model Metrics:", complex_metrics)
        print("Best Parameters:", self.best_params)
        
        # Generate predictions using the complex model
        print("\nGenerating Predictions...")
        leaderboard = self.generate_predictions(data, self.complex_model)
        
        return {
            'simple_metrics': simple_metrics,
            'complex_metrics': complex_metrics,
            'leaderboard': leaderboard
        }

# Usage example:
"""
# Assuming you already have your DataFrame loaded as 'data'
predictor = HomeRunPredictor()
results = predictor.run_full_pipeline(data)

# Save the leaderboard
results['leaderboard'].to_csv('Home_Run_Leader_Board.csv', index=False)

# Print top 30 predictions
print("\nTop 30 Home Run Predictions for 2025:")
print(results['leaderboard'].head(30))
"""