import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning for snippets.
    - Lowercase
    - Remove extra whitespace
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text
