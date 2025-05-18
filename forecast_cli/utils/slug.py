import re

def slugify_game(name: str) -> str:
    n = str(name) # Ensure it's a string, though type hint suggests it is
    n = n.lower()
    n = re.sub(r'[\s:\-\.\(\)/]+', '_', n) # Replace common separators, including dot, parens
    n = re.sub(r'[^0-9a-z_]', '', n)    # Remove anything not lowercase alphanum or underscore
    return re.sub(r'_+', '_', n).strip('_') # Collapse multiple underscores and strip ends 