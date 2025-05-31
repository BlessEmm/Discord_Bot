def convert_moneyline_to_implied_probability(odds: int) -> float:
    """
    Converts American moneyline odds to implied probability.

    Args:
        odds: American moneyline odds (e.g., -150, +120).

    Returns:
        Implied probability (0.0 to 1.0).
    """
    if odds < 0:  # Favorite
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:  # Underdog
        return 100 / (odds + 100)
    else:  # Even odds, though rare in final lines, can be represented as +100
        # Or handle as an error/special case if 0 is not expected.
        # For simplicity, let's assume odds are non-zero.
        # If odds can be 0, clarification on how to handle it would be needed.
        # Typically, even money is +100.
        raise ValueError("Odds of 0 are ambiguous. Use positive or negative values.")

def calculate_edge(model_prob: float, vegas_implied_prob: float) -> float:
    """
    Calculates the edge between the model's win probability and Vegas implied probability.

    Args:
        model_prob: The model's predicted win probability for a team.
        vegas_implied_prob: The implied win probability from Vegas odds for the same team.

    Returns:
        The calculated edge.
    """
    return model_prob - vegas_implied_prob
