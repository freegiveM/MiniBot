from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formatter import normalize_name


class FormatterTests(unittest.TestCase):
    def test_normalize_name_trims_and_slugifies_spaces(self):
        self.assertEqual(normalize_name(" Mini Bot "), "mini-bot")

    def test_normalize_name_preserves_existing_hyphen(self):
        self.assertEqual(normalize_name("Mini-Bot"), "mini-bot")


if __name__ == "__main__":
    unittest.main()

