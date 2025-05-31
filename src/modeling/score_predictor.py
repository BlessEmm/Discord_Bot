import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import numpy as np

class ScorePredictor:
    """
    A model to predict the final scores for MLB games using Gradient Boosting Regressors.
    Two separate models are trained: one for home team score and one for away team score.
    """
    def __init__(self, model_path: str = None, gbr_params: dict = None):
        """
        Initializes the ScorePredictor.

        Args:
            model_path (str, optional): Path to trained models. If provided, models are loaded.
            gbr_params (dict, optional): Parameters for GradientBoostingRegressor.
                                         Defaults to basic params.
        """
        self.home_score_model = None
        self.away_score_model = None
        self.feature_names_ = None # To store feature names used during training

        if gbr_params is None:
            self.gbr_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'subsample': 0.8,
                'random_state': 42
            }
        else:
            self.gbr_params = gbr_params

        if model_path:
            self.load_model(model_path)

    def train(self,
              training_data: pd.DataFrame,
              feature_columns: list,
              home_target_column: str = 'home_team_actual_score',
              away_target_column: str = 'away_team_actual_score'):
        """
        Trains the score prediction models (one for home team, one for away team).

        Args:
            training_data (pd.DataFrame): DataFrame containing features and target variables.
            feature_columns (list): List of column names to be used as features.
            home_target_column (str): Name of the column for home team's score.
            away_target_column (str): Name of the column for away team's score.
        """
        print(f"Training ScorePredictor (Gradient Boosting) models with {len(training_data)} records.")
        print(f"Features: {feature_columns[:5]}... (total {len(feature_columns)})")
        print(f"Home team target: '{home_target_column}', Away team target: '{away_target_column}'.")

        self.feature_names_ = feature_columns # Store feature names
        X_train = training_data[self.feature_names_]

        # Home Score Model
        y_home_train = training_data[home_target_column]
        # Drop rows where target is NaN, as GBR cannot handle NaN targets
        valid_home_indices = y_home_train.dropna().index
        X_train_home = X_train.loc[valid_home_indices]
        y_home_train_cleaned = y_home_train.loc[valid_home_indices]

        if not X_train_home.empty:
            print(f"Training home score model on {len(X_train_home)} records.")
            self.home_score_model = GradientBoostingRegressor(**self.gbr_params)
            self.home_score_model.fit(X_train_home, y_home_train_cleaned)
            print("Home score model training complete.")
        else:
            print("No valid data for home score model training after NaN removal.")


        # Away Score Model
        y_away_train = training_data[away_target_column]
        # Drop rows where target is NaN
        valid_away_indices = y_away_train.dropna().index
        X_train_away = X_train.loc[valid_away_indices]
        y_away_train_cleaned = y_away_train.loc[valid_away_indices]

        if not X_train_away.empty:
            print(f"Training away score model on {len(X_train_away)} records.")
            self.away_score_model = GradientBoostingRegressor(**self.gbr_params)
            self.away_score_model.fit(X_train_away, y_away_train_cleaned)
            print("Away score model training complete.")
        else:
            print("No valid data for away score model training after NaN removal.")

        print("ScorePredictor model training finished.")


    def predict_score(self, game_features: pd.DataFrame) -> dict:
        """
        Predicts the scores for both home and away teams for one or more games.

        Args:
            game_features (pd.DataFrame): DataFrame containing features for the game(s).
                                         Columns must match feature_columns from training.

        Returns:
            dict or list[dict]:
                - If single game (1 row DF): Dict e.g., {'predicted_home_score': 4.5, 'predicted_away_score': 3.2}.
                - If multiple games: A list of such dictionaries.
        """
        if not self.home_score_model or not self.away_score_model:
            raise ValueError("Models have not been trained or loaded. Call train() or load_model() first.")

        if self.feature_names_ is None:
            raise ValueError("Feature names not set. Model might be loaded from an older version or not trained.")

        if not all(f in game_features.columns for f in self.feature_names_):
            missing_cols = [f for f in self.feature_names_ if f not in game_features.columns]
            raise ValueError(f"Missing feature columns in input: {missing_cols}. Expected: {self.feature_names_}")

        # Ensure column order is the same as during training
        game_features_aligned = game_features[self.feature_names_]

        predicted_home_scores = self.home_score_model.predict(game_features_aligned)
        predicted_away_scores = self.away_score_model.predict(game_features_aligned)

        # Ensure scores are not negative (common post-processing for score prediction)
        predicted_home_scores = np.maximum(0, predicted_home_scores)
        predicted_away_scores = np.maximum(0, predicted_away_scores)

        results = []
        if isinstance(predicted_home_scores, float): # Single prediction (though predict usually returns array)
             predicted_home_scores = [predicted_home_scores]
             predicted_away_scores = [predicted_away_scores] # Should also be single if home is

        for i in range(len(predicted_home_scores)):
            results.append({
                'predicted_home_score': round(predicted_home_scores[i], 2),
                'predicted_away_score': round(predicted_away_scores[i], 2)
            })

        return results[0] if len(results) == 1 else results


    def save_model(self, path: str):
        """
        Saves the trained score prediction models (home and away) and feature names.
        Models are saved in a dictionary to a single file using joblib.
        """
        if not self.home_score_model or not self.away_score_model:
            print("No models to save.")
            return

        print(f"Saving ScorePredictor models to {path}")
        model_artifact = {
            'home_score_model': self.home_score_model,
            'away_score_model': self.away_score_model,
            'feature_names': self.feature_names_,
            'gbr_params': self.gbr_params
        }
        joblib.dump(model_artifact, path)
        print(f"Models saved to {path}")

    def load_model(self, path: str):
        """
        Loads trained score prediction models (home and away) and feature names.
        """
        print(f"Loading ScorePredictor models from {path}")
        try:
            model_artifact = joblib.load(path)
            self.home_score_model = model_artifact['home_score_model']
            self.away_score_model = model_artifact['away_score_model']
            self.feature_names_ = model_artifact.get('feature_names') # Use .get for backward compatibility
            self.gbr_params = model_artifact.get('gbr_params', self.gbr_params) # Load params or keep default

            if self.feature_names_ is None:
                print("Warning: Feature names not found in loaded model artifact. Ensure prediction input matches training features.")
            else:
                print(f"  Models trained with {len(self.feature_names_)} features. First few: {self.feature_names_[:5]}")
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.home_score_model = None
            self.away_score_model = None
            self.feature_names_ = None
            raise e

    # def _preprocess_features(self, game_features: pd.Series) -> pd.DataFrame:
    #     # Placeholder, actual preprocessing should happen before calling train/predict.
    #     return pd.DataFrame([game_features])
