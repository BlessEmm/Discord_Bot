import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold # For CV in CalibratedClassifierCV
import joblib

class WinPredictor:
    """
    A model to predict the win probability for MLB games using XGBoost,
    with probability calibration using CalibratedClassifierCV.
    """
    def __init__(self, model_path: str = None, xgb_params: dict = None, calibration_method: str = 'isotonic', cv_folds: int = 3):
        """
        Initializes the WinPredictor.

        Args:
            model_path (str, optional): Path to a trained calibrated model. Defaults to None.
            xgb_params (dict, optional): XGBoost parameters for the base estimator.
            calibration_method (str, optional): 'isotonic' or 'sigmoid'. Defaults to 'isotonic'.
            cv_folds (int, optional): Number of folds for CalibratedClassifierCV. Defaults to 3.
        """
        self.calibrated_model = None # This will be the CalibratedClassifierCV instance
        self.feature_names_ = None # Store feature names

        if xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                # 'eval_metric': ['logloss', 'auc'], # eval_metric for xgb.train, not directly for XGBClassifier in CalibratedCV
                'eta': 0.1,
                'max_depth': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'seed': 42,
                'n_estimators': 100 # XGBClassifier uses n_estimators
            }
        else:
            self.xgb_params = xgb_params
            if 'n_estimators' not in self.xgb_params: # Ensure n_estimators is present for XGBClassifier
                self.xgb_params['n_estimators'] = 100

        self.calibration_method = calibration_method
        self.cv_folds = cv_folds

        if model_path:
            self.load_model(model_path)

    def train(self, training_data: pd.DataFrame, feature_columns: list, target_column: str = 'home_team_win'):
        """
        Trains the calibrated win prediction model.
        The XGBoost model is used as a base estimator for CalibratedClassifierCV.

        Args:
            training_data (pd.DataFrame): DataFrame containing features and the target variable.
            feature_columns (list): List of column names to be used as features.
            target_column (str): Name of the column representing the win/loss label (1 for home win, 0 for away win).
        """
        print(f"Training Calibrated WinPredictor (XGBoost base) with {len(training_data)} records on target '{target_column}'.")
        print(f"Features: {feature_columns[:5]}... (total {len(feature_columns)})")
        print(f"Using XGBoost base parameters: {self.xgb_params}")
        print(f"Calibration method: {self.calibration_method}, CV folds: {self.cv_folds}")

        self.feature_names_ = feature_columns
        X_train = training_data[self.feature_names_]
        y_train = training_data[target_column]

        # Initialize XGBClassifier (SKLearn wrapper for XGBoost)
        # Note: Some params like 'eta' are 'learning_rate' in XGBClassifier
        xgb_clf_params = self.xgb_params.copy()
        if 'eta' in xgb_clf_params:
            xgb_clf_params['learning_rate'] = xgb_clf_params.pop('eta')
        # eval_metric can be set during fit if early stopping is used, but CalibratedClassifierCV handles fitting.
        if 'eval_metric' in xgb_clf_params:
             del xgb_clf_params['eval_metric'] # Not directly used by XGBClassifier constructor in this way

        base_clf = xgb.XGBClassifier(**xgb_clf_params)

        # Initialize CalibratedClassifierCV
        # StratifiedKFold is good for classification tasks to preserve class proportions in folds.
        cv_strategy = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.xgb_params.get('seed', 42))

        self.calibrated_model = CalibratedClassifierCV(
            estimator=base_clf, # In scikit-learn >= 1.4, base_estimator is deprecated, use estimator
            method=self.calibration_method,
            cv=cv_strategy # Can also be an integer for (Stratified)KFold, or a CV splitter
        )

        print("Fitting CalibratedClassifierCV...")
        self.calibrated_model.fit(X_train, y_train)

        print("Calibrated WinPredictor training complete.")

    def predict_proba(self, game_features: pd.DataFrame) -> dict:
        """
        Predicts the calibrated win probability for a single game or multiple games.

        Args:
            game_features (pd.DataFrame): DataFrame containing features for one or more games.
                                         Columns should match the feature_columns used during training.

        Returns:
            dict or list[dict]:
                - If single game (1 row DF): A dictionary e.g., {'home_win_prob': 0.65, 'away_win_prob': 0.35}.
                - If multiple games: A list of such dictionaries.
        """
        if self.calibrated_model is None:
            raise ValueError("Model has not been trained or loaded. Call train() or load_model() first.")

        if self.feature_names_ is None:
             raise ValueError("Feature names not set. Model might not have been trained or loaded correctly.")

        if not all(f in game_features.columns for f in self.feature_names_):
            missing_cols = [f for f in self.feature_names_ if f not in game_features.columns]
            raise ValueError(f"Missing feature columns in input: {missing_cols}. Expected: {self.feature_names_}")

        game_features_aligned = game_features[self.feature_names_]

        # predict_proba returns probabilities for both classes [prob_class_0, prob_class_1]
        # Assuming class 1 is 'home_team_win'
        calibrated_probs_array = self.calibrated_model.predict_proba(game_features_aligned)

        results = []
        for i in range(calibrated_probs_array.shape[0]):
            home_win_prob = calibrated_probs_array[i, 1] # Probability of the positive class (home_team_win)
            results.append({'home_win_prob': home_win_prob, 'away_win_prob': 1.0 - home_win_prob})

        return results[0] if len(results) == 1 else results

    def save_model(self, path: str):
        """ Saves the trained CalibratedClassifierCV model. """
        if self.calibrated_model is None:
            print("No model to save.")
            return

        print(f"Saving Calibrated WinPredictor model to {path}")
        model_artifact = {
            'calibrated_model': self.calibrated_model,
            'feature_names': self.feature_names_,
            'xgb_params': self.xgb_params, # Save original xgb_params for reference
            'calibration_method': self.calibration_method,
            'cv_folds': self.cv_folds
        }
        joblib.dump(model_artifact, path)
        print(f"Model saved to {path}.")

    def load_model(self, path: str):
        """ Loads a trained CalibratedClassifierCV model. """
        print(f"Loading Calibrated WinPredictor model from {path}")
        try:
            model_artifact = joblib.load(path)
            self.calibrated_model = model_artifact['calibrated_model']
            self.feature_names_ = model_artifact.get('feature_names')
            self.xgb_params = model_artifact.get('xgb_params')
            self.calibration_method = model_artifact.get('calibration_method')
            self.cv_folds = model_artifact.get('cv_folds')

            if not all([self.feature_names_, self.xgb_params, self.calibration_method, self.cv_folds]):
                 print("Warning: Some metadata (feature_names, xgb_params, etc.) missing from loaded model. Defaults may be used.")

            print(f"Calibrated model loaded successfully from {path}.")
            if self.feature_names_:
                 print(f"  Model trained with {len(self.feature_names_)} features. First few: {self.feature_names_[:5]}")

        except Exception as e:
            print(f"Error loading calibrated model: {e}")
            self.calibrated_model = None
            raise e
