"""
Utility functions for data ingestion
"""

import sys


def safe_print(text: str):
    """
    Safely print text with Unicode characters, falling back to ASCII if needed

    Args:
        text: Text to print
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: remove emojis and special characters
        ascii_text = text.encode('ascii', 'ignore').decode('ascii')
        print(ascii_text)


# Emoji mappings (with ASCII fallback)
EMOJI = {
    'rocket': '\U0001F680' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '>',
    'bus': '\U0001F68C' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[BUS]',
    'crystal_ball': '\U0001F52E' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '*',
    'check': '\u2713' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else 'OK',
    'timer': '\u23F1' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[TIME]',
    'floppy': '\U0001F4BE' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[SAVE]',
    'folder': '\U0001F4C1' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[FOLDER]',
    'chart': '\U0001F4CA' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[CHART]',
    'location': '\U0001F4CD' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[LOC]',
    'weather': '\U0001F326' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[WEATHER]',
    'party': '\U0001F389' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[EVENT]',
    'search': '\U0001F50D' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '[SEARCH]',
    'hourglass': '\u231B' if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower() else '...',
}
