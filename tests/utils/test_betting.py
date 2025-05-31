import unittest
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utils.betting import convert_moneyline_to_implied_probability

class TestBettingUtils(unittest.TestCase):

    def test_convert_moneyline_to_implied_probability_favorite(self):
        """Test conversion for favorite (negative odds)."""
        # Examples:
        # -110 should be 110 / (110 + 100) = 110 / 210 = 0.5238
        # -200 should be 200 / (200 + 100) = 200 / 300 = 0.6666
        # -150 should be 150 / (150 + 100) = 150 / 250 = 0.6
        self.assertAlmostEqual(convert_moneyline_to_implied_probability(-110), 110/210, places=4)
        self.assertAlmostEqual(convert_moneyline_to_implied_probability(-200), 200/300, places=4)
        self.assertAlmostEqual(convert_moneyline_to_implied_probability(-150), 150/250, places=4)

    def test_convert_moneyline_to_implied_probability_underdog(self):
        """Test conversion for underdog (positive odds)."""
        # Examples:
        # +100 should be 100 / (100 + 100) = 100 / 200 = 0.5
        # +150 should be 100 / (150 + 100) = 100 / 250 = 0.4
        # +200 should be 100 / (200 + 100) = 100 / 300 = 0.3333
        self.assertAlmostEqual(convert_moneyline_to_implied_probability(100), 100/200, places=4)
        self.assertAlmostEqual(convert_moneyline_to_implied_probability(150), 100/250, places=4)
        self.assertAlmostEqual(convert_moneyline_to_implied_probability(200), 100/300, places=4)

    def test_convert_moneyline_to_implied_probability_edge_cases(self):
        """Test edge cases like very large odds or small odds (close to even)."""
        self.assertAlmostEqual(convert_moneyline_to_implied_probability(-500), 500/600, places=4) # Strong favorite
        self.assertAlmostEqual(convert_moneyline_to_implied_probability(500), 100/600, places=4) # Strong underdog
        self.assertAlmostEqual(convert_moneyline_to_implied_probability(-101), 101/201, places=4) # Close to even
        self.assertAlmostEqual(convert_moneyline_to_implied_probability(101), 100/201, places=4) # Close to even

    def test_convert_moneyline_to_implied_probability_invalid_input(self):
        """Test invalid input (odds of 0)."""
        with self.assertRaises(ValueError):
            convert_moneyline_to_implied_probability(0)

if __name__ == '__main__':
    unittest.main()
