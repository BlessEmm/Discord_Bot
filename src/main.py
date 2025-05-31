import pandas as pd
from datetime import datetime
import numpy as np
import os # Ensure os is imported

# Import model classes
from modeling.win_predictor import WinPredictor
from modeling.score_predictor import ScorePredictor
from utils.betting import convert_moneyline_to_implied_probability, calculate_edge
from data_processing.api_clients import fetch_live_mlb_odds

# --- Configuration for Models (for easier tuning simulation) ---
WIN_PREDICTOR_XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'eta': 0.05, # Example: Adjusted learning rate
    'max_depth': 4, # Example: Adjusted max_depth
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'seed': 42
}
WIN_PREDICTOR_NUM_BOOST_ROUND = 75 # Example: Adjusted rounds

SCORE_PREDICTOR_GBR_PARAMS = {
    'n_estimators': 120, # Example: Adjusted n_estimators
    'learning_rate': 0.05, # Example: Adjusted learning_rate
    'max_depth': 4, # Example: Adjusted max_depth
    'subsample': 0.7,
    'random_state': 42
}

# --- Placeholder Data Generation (from previous step, unchanged) ---
def generate_dummy_processed_data(num_rows=200) -> pd.DataFrame:
    print(f"Generating {num_rows} rows of dummy processed data for demonstration...")
    data = {
        'game_id': range(num_rows),
        'date': pd.to_datetime([datetime(2023, 4, 1) + pd.Timedelta(days=i) for i in range(num_rows)]),
        'home_team_id': np.random.choice(['NYY', 'BOS', 'LAA', 'HOU', 'TOR', 'ATL'], size=num_rows),
        'away_team_id': np.random.choice(['NYM', 'PHI', 'OAK', 'SEA', 'BAL', 'MIA'], size=num_rows),
        'home_avg_runs_last_10': np.random.normal(4.5, 1, num_rows).round(2),
        'away_avg_runs_last_10': np.random.normal(4.2, 1, num_rows).round(2),
        'home_sp_era': np.random.normal(3.8, 0.5, num_rows).round(2),
        'away_sp_era': np.random.normal(4.1, 0.5, num_rows).round(2),
        'home_bullpen_era': np.random.normal(3.5, 0.3, num_rows).round(2),
        'away_bullpen_era': np.random.normal(3.6, 0.3, num_rows).round(2),
        'park_factor_runs': np.random.normal(1.0, 0.05, num_rows).round(3),
        'home_win_streak': np.random.randint(-3, 4, size=num_rows),
        'away_travel_days': np.random.randint(0, 3, size=num_rows),
        'is_division_game': np.random.choice([0,1], size=num_rows),
        'home_team_actual_score': np.random.randint(0, 10, size=num_rows),
        'away_team_actual_score': np.random.randint(0, 10, size=num_rows),
    }
    df = pd.DataFrame(data)
    df['home_team_win'] = (df['home_team_actual_score'] > df['away_team_actual_score']).astype(int)
    df['home_moneyline_odds'] = np.random.randint(-250, -105, size=num_rows)
    df['away_moneyline_odds'] = np.random.randint(100, 200, size=num_rows)
    print("Dummy data generation complete.")
    return df

# --- Main Orchestration (Modified for Iteration Demonstration) ---
def main():
    print("Starting MLB Betting Model Orchestration (Iteration Demo)...")

    # Create models directory if it doesn't exist (moved here for clarity)
    os.makedirs("models", exist_ok=True)

    print("\n--- Phase 1: Data Loading and Preparation (Dummy) ---")
    raw_df = generate_dummy_processed_data(num_rows=500)
    feature_columns = [
        'home_avg_runs_last_10', 'away_avg_runs_last_10', 'home_sp_era', 'away_sp_era',
        'home_bullpen_era', 'away_bullpen_era', 'park_factor_runs', 'home_win_streak',
        'away_travel_days', 'is_division_game'
    ]
    split_index = int(len(raw_df) * 0.8)
    train_df = raw_df.iloc[:split_index].copy()
    test_df = raw_df.iloc[split_index:].copy()
    print(f"Training data: {len(train_df)} rows, Test data: {len(test_df)} rows")
    print(f"Feature columns for models: {feature_columns}")

    print("\n--- Phase 2: Model Initialization (with example tuned params) ---")
    win_predictor = WinPredictor(xgb_params=WIN_PREDICTOR_XGB_PARAMS)
    score_predictor = ScorePredictor(gbr_params=SCORE_PREDICTOR_GBR_PARAMS)

    print("\n--- Phase 3: Model Training (with Dummy Data & new params) ---")
    print("Training WinPredictor...")
    win_predictor.train(
        training_data=train_df, feature_columns=feature_columns,
        target_column='home_team_win', num_boost_round=WIN_PREDICTOR_NUM_BOOST_ROUND
    )
    win_predictor.save_model("models/dummy_win_predictor_iterated.joblib")

    print("\nTraining ScorePredictor...")
    score_predictor.train(
        training_data=train_df, feature_columns=feature_columns,
        home_target_column='home_team_actual_score', away_target_column='away_team_actual_score'
    )
    score_predictor.save_model("models/dummy_score_predictor_iterated.joblib")

    print("\n--- Phase 4: Prediction on Test Set (Dummy Data) ---")
    print("
--- Conceptual: Fetching Live Odds for Current Day Games ---")
    # live_odds_data = fetch_live_mlb_odds(api_key='YOUR_ODDS_API_KEY') # Replace with actual key management
    # if live_odds_data:
    #     print(f"Fetched {len(live_odds_data)} live games with odds.")
    # else:
    #     print("No live odds fetched or an error occurred.")
    if test_df.empty or not all(f in test_df.columns for f in feature_columns):
        print("Test data is empty or missing feature columns. Skipping prediction.")
    else:
        X_test_win = test_df[feature_columns]
        win_predictions_list = win_predictor.predict_proba(X_test_win)
        test_df['model_home_win_prob'] = [p['home_win_prob'] for p in win_predictions_list]
        test_df['model_away_win_prob'] = [p['away_win_prob'] for p in win_predictions_list]
        print("Win probabilities predicted.")

        X_test_score = test_df[feature_columns]
        score_predictions_list = score_predictor.predict_score(X_test_score)
        test_df['predicted_home_score'] = [p['predicted_home_score'] for p in score_predictions_list]
        test_df['predicted_away_score'] = [p['predicted_away_score'] for p in score_predictions_list]
        print("Scores predicted.")

    print("\n--- Phase 5: Model Evaluation (Placeholder Metrics) ---")
    # (Value Pick Identification section removed for brevity in this refactoring focused on training iteration)
    # (Actual evaluation metrics using sklearn.metrics would be added in a real scenario)
    if not test_df.empty and 'model_home_win_prob' in test_df.columns:
        actual_wins = test_df['home_team_win']
        predicted_probs = test_df['model_home_win_prob']
        print("  Win Predictor Evaluation:")
        print(f"    LogLoss: N/A ") # Placeholder
        print(f"    AUC-ROC: N/A ") # Placeholder
        print("    Feature Importances (XGBoost):")
        if hasattr(win_predictor.model, 'get_score'): # Check if model is trained and has get_score (XGBoost)
             f_importance = win_predictor.model.get_score(importance_type='gain')
             print(f"      {f_importance}")
        else:
            print("      Could not retrieve feature importance for WinPredictor (model not available or not XGBoost type).")

        print("\n  Score Predictor Evaluation:")
        # Placeholder for score metrics
        if 'predicted_home_score' in test_df.columns:
             print(f"    Home Score MAE: N/A") # Placeholder
             print(f"    Away Score MAE: N/A") # Placeholder

        print("    Feature Importances (Home Score GBR):")
        if score_predictor.home_score_model and hasattr(score_predictor.home_score_model, 'feature_importances_'):
            home_imp = dict(zip(score_predictor.feature_names_, score_predictor.home_score_model.feature_importances_))
            print(f"      {home_imp}")
        else:
            print("      Could not retrieve feature importance for Home Score model.")

        print("    Feature Importances (Away Score GBR):")
        if score_predictor.away_score_model and hasattr(score_predictor.away_score_model, 'feature_importances_'):
            away_imp = dict(zip(score_predictor.feature_names_, score_predictor.away_score_model.feature_importances_))
            print(f"      {away_imp}")
        else:
            print("      Could not retrieve feature importance for Away Score model.")
    else:
        print("  Skipping evaluation as test predictions are not available.")

    print("\nMLB Betting Model Orchestration (Iteration Demo) Complete.")

if __name__ == '__main__':
    main()
