"""Collects plain-English status messages that get shown on the website."""
MESSAGES: list[str] = []


def note(msg: str):
    print("  " + msg)
    if msg not in MESSAGES:
        MESSAGES.append(msg)
