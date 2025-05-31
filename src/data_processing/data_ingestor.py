import os
import json
import time
from .api_clients import fetch_game_schedule, fetch_game_data

RAW_DATA_DIR = "data/raw"

def ingest_season_data(season: int, start_date: str = None, end_date: str = None):
    """
    Orchestrates the fetching of game data for a given season and saves it.
    Optionally fetches data for a specific date range within the season.

    Args:
        season (int): The year of the season.
        start_date (str, optional): YYYY-MM-DD. Fetches games on or after this date.
        end_date (str, optional): YYYY-MM-DD. Fetches games on or before this date.
    """
    print(f"Starting data ingestion for season {season}...")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Note: The current fetch_game_schedule gets ALL gamePks for the season.
    # Filtering by date would ideally happen at the API call level if the API supports it easily,
    # or by post-filtering the schedule results.
    # For simplicity, we fetch all and then could filter before fetching details.
    # The MLB Stats API schedule endpoint can take startDate and endDate parameters.
    # Let's assume api_clients.fetch_game_schedule is updated or we filter here.

    all_game_pks = fetch_game_schedule(season) # This needs to be adapted if date filtering is strict for schedule

    # Placeholder: If start_date/end_date are provided, we'd ideally filter game_pks here
    # based on game date from the schedule data before fetching details.
    # This requires fetch_game_schedule to return more than just pks (e.g., game dates too).
    # For now, this example will fetch details for all pks returned by fetch_game_schedule.

    game_pks_to_process = all_game_pks

    if not game_pks_to_process:
        print(f"No gamePks found for season {season}. Ingestion halted.")
        return

    season_data_dir = os.path.join(RAW_DATA_DIR, str(season))
    os.makedirs(season_data_dir, exist_ok=True)

    total_games = len(game_pks_to_process)
    print(f"Found {total_games} games to process for season {season}.")

    for i, game_pk in enumerate(game_pks_to_process):
        print(f"Processing game {i+1}/{total_games}, gamePk: {game_pk}")

        # Check if data already exists to avoid re-fetching (simple check)
        game_file_path = os.path.join(season_data_dir, f"{game_pk}.json")
        if os.path.exists(game_file_path):
            print(f"  Data for gamePk {game_pk} already exists. Skipping.")
            continue

        game_detail = fetch_game_data(game_pk)

        if game_detail.get("liveFeed") or game_detail.get("boxscore"): # Save if at least some data was fetched
            with open(game_file_path, 'w') as f:
                json.dump(game_detail, f, indent=4)
            print(f"  Successfully fetched and saved data for gamePk {game_pk} to {game_file_path}")
        else:
            print(f"  Failed to fetch significant data for gamePk {game_pk}. Not saving.")

        # Respectful delay
        time.sleep(0.5) # Increased delay between fetching full game details

    print(f"Data ingestion for season {season} complete.")

if __name__ == '__main__':
    # Example: Ingest data for a small part of a recent season
    # Be mindful of API rate limits and the volume of data for a full season.
    # It's often better to run this for specific date ranges or a limited number of games initially.

    # To run this, you'd typically do: python -m src.data_processing.data_ingestor
    # The example below is for conceptual testing.

    # test_ingest_season = 2023
    # print(f"Example: Ingesting data for season {test_ingest_season}")
    # print("NOTE: This will attempt to fetch data for the entire season if not date-restricted.")
    # print("Consider fetching for a very small date range or a specific game for testing.")
    # ingest_season_data(test_ingest_season)

    # Example of how one might call it for a specific game (after modifying ingest_season_data or adding new func):
    # game_pk_to_test = "718281" # Example gamePk from 2023
    # game_detail = fetch_game_data(game_pk_to_test)
    # if game_detail:
    #     os.makedirs(os.path.join(RAW_DATA_DIR, "test_game"), exist_ok=True)
    #     with open(os.path.join(RAW_DATA_DIR, "test_game", f"{game_pk_to_test}.json"), 'w') as f:
    #         json.dump(game_detail, f, indent=4)
    #     print(f"Saved test data for gamePk {game_pk_to_test}")
    pass
