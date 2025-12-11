import sys
import os

# Ensure the project root is on sys.path so we can import src.preprocessing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_text


def test_clean_text_basic():
    """
    Basic test: HTML tags and non-alphabetic characters should be removed,
    and the text should be converted to lowercase.
    """
    raw_text = "Hello <br> World! 123"
    cleaned = clean_text(raw_text)

    # Normalize spaces to avoid double-space issues
    normalized = " ".join(cleaned.split())
    assert normalized == "hello world"


def test_clean_text_no_change():
    """
    Text that is already clean and lowercase should remain unchanged.
    """
    text = "pure text"
    cleaned = clean_text(text)
    assert cleaned == "pure text"