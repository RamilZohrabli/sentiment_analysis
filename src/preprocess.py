import re

_whitespace_re = re.compile(r"\s+")
_digits_re = re.compile(r"\d+")

def clean_text(text: str) -> str:
    """
    Preprocessing includes
    - lowercase
    - remove digits
    - normalize whitespace
    """
    if text is None:
        return ""

    text = str(text)
    text = text.lower()
    text = _digits_re.sub("", text)
    text = _whitespace_re.sub(" ", text).strip()

    return text
