import pandas as pd
import json
import os

# This will be expanded significantly as we understand the data and model needs.

def load_raw_game_data(game_pk: str, season: int, raw_data_dir: str = "data/raw") -> dict:
    """Loads raw JSON data for a single game."""
    file_path = os.path.join(raw_data_dir, str(season), f"{game_pk}.json")
    if not os.path.exists(file_path):
        print(f"Raw data file not found: {file_path}")
        return None
    with open(file_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path}")
            return None

def extract_basic_game_info(raw_game_data: dict, game_pk: str) -> dict:
    """
    Placeholder function to extract very basic info from the raw game data.
    This will need to be much more sophisticated.
    """
    if not raw_game_data or not raw_game_data.get("liveFeed") or not raw_game_data.get("boxscore"):
        print(f"Missing critical data for game_pk {game_pk} in extract_basic_game_info")
        return None

    live_data = raw_game_data["liveFeed"].get("gameData")
    box_data = raw_game_data["boxscore"].get("teams")
    meta_data = raw_game_data["liveFeed"].get("metaData", {})

    if not live_data or not box_data:
        print(f"Missing gameData or teams in boxscore for game_pk {game_pk}")
        return None

    info = {}
    info['game_pk'] = game_pk
    info['game_date'] = live_data.get("datetime", {}).get("dateTime")
    info['season'] = live_data.get("game", {}).get("season")

    info['home_team_id'] = live_data.get("teams", {}).get("home", {}).get("abbreviation", "N/A")
    info['away_team_id'] = live_data.get("teams", {}).get("away", {}).get("abbreviation", "N/A")

    # Scores (from boxscore, could also get from linescore)
    if box_data and "home" in box_data and "away" in box_data:
        info['home_score'] = box_data["home"].get("teamStats", {}).get("batting", {}).get("runs", None)
        info['away_score'] = box_data["away"].get("teamStats", {}).get("batting", {}).get("runs", None)
    else:
        info['home_score'] = None
        info['away_score'] = None

    info['venue_name'] = live_data.get("venue", {}).get("name", "N/A")
    info['game_status'] = live_data.get("status", {}).get("detailedState", "N/A")

    # Example: Did home team win? (requires game to be final)
    if info['game_status'] == "Final" and info['home_score'] is not None and info['away_score'] is not None:
        info['home_team_won'] = 1 if info['home_score'] > info['away_score'] else 0
    else:
        info['home_team_won'] = None

    # This is a tiny fraction of what's needed.
    # We'd need to parse player stats, pitcher stats, team stats up to that game, etc.

    return info

def process_season_into_dataframe(season: int, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
    """
    Loads all raw game data for a season, extracts basic features, and saves as a Parquet file.
    This is a very basic first pass.
    """
    print(f"Processing season {season} into a DataFrame...")
    season_raw_path = os.path.join(raw_data_dir, str(season))
    if not os.path.isdir(season_raw_path):
        print(f"Raw data directory for season {season} not found: {season_raw_path}")
        return

    all_game_features = []
    for fname in os.listdir(season_raw_path):
        if fname.endswith(".json"):
            game_pk = fname.split('.')[0]
            print(f"  Processing game_pk: {game_pk}")
            raw_data = load_raw_game_data(game_pk, season, raw_data_dir)
            if raw_data:
                features = extract_basic_game_info(raw_data, game_pk)
                if features:
                    all_game_features.append(features)

    if not all_game_features:
        print(f"No features extracted for season {season}. DataFrame not created.")
        return

    df = pd.DataFrame(all_game_features)
    os.makedirs(processed_data_dir, exist_ok=True)
    output_path = os.path.join(processed_data_dir, f"{season}_basic_gamelogs.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved basic processed data for season {season} to {output_path}")
    print(df.head())


if __name__ == '__main__':
    # Example usage:
    # test_season_process = 2023 # Assuming data for 2023 has been ingested into data/raw/2023
    # print(f"Example: Processing raw data for season {test_season_process}")
    # process_season_into_dataframe(test_season_process)
    pass
