import requests
import time

MLB_STATS_API_BASE_URL = "https://statsapi.mlb.com/api/v1"

def fetch_game_schedule(season: int) -> list:
    """
    Fetches the schedule for a given MLB season to get gamePks.

    Args:
        season (int): The year of the season (e.g., 2023).

    Returns:
        list: A list of gamePks for the season, or an empty list if an error occurs.
    """
    game_pks = []
    url = f"{MLB_STATS_API_BASE_URL}/schedule?sportId=1&season={season}&hydrate=team,linescore,flags,liveLookin,person,stats,probablePitcher,game(content(summary)),seriesStatus(series)"

    print(f"Fetching schedule for season: {season} from {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        if not data.get("dates"):
            print(f"No dates found in schedule data for season {season}.")
            return game_pks

        for date_entry in data["dates"]:
            for game in date_entry.get("games", []):
                if "gamePk" in game:
                    game_pks.append(game["gamePk"])

        print(f"Found {len(game_pks)} games for season {season}.")
        return game_pks

    except requests.exceptions.RequestException as e:
        print(f"Error fetching schedule for season {season}: {e}")
        return []
    except ValueError as e: # Handle JSON decoding errors
        print(f"Error decoding JSON for season {season}: {e}")
        return []

def fetch_game_data(game_pk: str) -> dict:
    """
    Fetches detailed game data for a given gamePk.
    This includes live feed (play-by-play), boxscore, and linecore.

    Args:
        game_pk (str): The unique identifier for the game.

    Returns:
        dict: A dictionary containing the game data, or an empty dict if an error occurs.
              Keys will be 'liveFeed', 'boxscore', 'linescore'.
    """
    game_data = {}

    # Endpoint for live feed (play-by-play, game events, etc.)
    live_feed_url = f"{MLB_STATS_API_BASE_URL}/game/{game_pk}/feed/live?language=en"
    # Endpoint for boxscore
    boxscore_url = f"{MLB_STATS_API_BASE_URL}/game/{game_pk}/boxscore"
    # Endpoint for linescore (often included in schedule, but can be fetched separately)
    # linescore_url = f"{MLB_STATS_API_BASE_URL}/game/{game_pk}/linescore" # Redundant if schedule hydration is good

    urls_to_fetch = {
        "liveFeed": live_feed_url,
        "boxscore": boxscore_url
    }

    print(f"Fetching detailed data for gamePk: {game_pk}")

    for key, url in urls_to_fetch.items():
        try:
            print(f"  Fetching {key} from {url}")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            game_data[key] = response.json()
            time.sleep(0.2) # Small delay to be polite to the API
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {key} for gamePk {game_pk}: {e}")
            game_data[key] = None # Store None if a part fails
        except ValueError as e: # Handle JSON decoding errors
            print(f"  Error decoding JSON for {key}, gamePk {game_pk}: {e}")
            game_data[key] = None

    return game_data


# --- Appended content for The Odds API ---

THE_ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

def fetch_live_mlb_odds(api_key: str, regions: str = 'us', markets: str = 'h2h', sport: str = 'baseball_mlb') -> list:
    """
    Fetches live MLB moneyline odds from The Odds API.

    Args:
        api_key (str): Your API key for The Odds API.
        regions (str, optional): Comma-separated list of regions (e.g., 'us', 'uk', 'au', 'eu'). Defaults to 'us'.
        markets (str, optional): Comma-separated list of markets (e.g., 'h2h', 'spreads', 'totals').
                                 'h2h' is for moneyline. Defaults to 'h2h'.
        sport (str, optional): The sport key. Defaults to 'baseball_mlb'.

    Returns:
        list: A list of dictionaries, where each dictionary contains details for a game and its odds.
              Returns an empty list if an error occurs or no games are found.
    """
    url = f"{THE_ODDS_API_BASE_URL}/sports/{sport}/odds"
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': markets,
        'oddsFormat': 'american' # For -150, +130 style odds
    }

    print(f"Fetching live MLB odds from The Odds API: {url}")
    print(f"Params: regions={regions}, markets={markets}")

    try:
        response = requests.get(url, params=params, timeout=20) # Increased timeout for external API
        response.raise_for_status() # Raise an exception for HTTP errors

        odds_data = response.json()

        if not odds_data:
            print("No upcoming games found or API returned empty data.")
            return []

        print(f"Odds API Usage: {response.headers.get('X-Requests-Remaining')} requests remaining.")
        print(f"Odds API Usage: Used {response.headers.get('X-Requests-Used')} requests.")
        return odds_data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching live odds: {e}")
        return []
    except ValueError as e: # Handle JSON decoding errors
        print(f"Error decoding JSON from The Odds API: {e}")
        return []

if __name__ == '__main__':
    # (Previous test code for fetch_game_schedule and fetch_game_data might be here)
    # Ensure MLB Stats API tests are minimal if they exist or comment them out for this specific test.
    # print("Skipping MLB Stats API tests for this run...")


    print("\n--- Testing The Odds API client function ---")
    api_key_to_test = "3c8c0497dfaf1abba28037ea75f1994d" # This will be replaced by the actual key by shell

    if not api_key_to_test or api_key_to_test == "YOUR_API_KEY_HERE": # Basic check
        print("The Odds API key not found or is a placeholder. Skipping live odds fetch test.")
    else:
        print(f"Attempting to fetch live MLB odds with key: {api_key_to_test[:4]}... (masked)")
        live_odds = fetch_live_mlb_odds(api_key_to_test)
        if live_odds:
            print(f"Successfully fetched {len(live_odds)} upcoming game odds from The Odds API.")
            if len(live_odds) > 0:
                first_game = live_odds[0]
                print(f"  Example Game: {first_game.get('home_team')} vs {first_game.get('away_team')} at {first_game.get('commence_time')}")
                if first_game.get('bookmakers'):
                    bookie = first_game['bookmakers'][0]
                    print(f"    Bookmaker: {bookie.get('title')}")
                    for market in bookie.get('markets', []):
                        if market.get('key') == 'h2h':
                            for outcome in market.get('outcomes', []):
                                print(f"      {outcome['name']}: {outcome['price']}")
                else:
                    print("    No bookmaker odds found in the first game's data.")
        else:
            print("Failed to fetch live odds or no games available.")
    print("--- The Odds API client function testing complete ---")
