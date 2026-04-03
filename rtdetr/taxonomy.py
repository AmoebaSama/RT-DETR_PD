from __future__ import annotations

RAW_TO_DISPLAY = {
    "good": "Good Solder",
    "exc_solder": "Excess Solder",
    "poor_solder": "Insufficient Solder",
    "spike": "Solder Spike",
    "no_good": "Defective Solder",
}

SUPPORTED_REQUESTED_CLASSES = [
    "Good Solder",
    "Excess Solder",
    "Insufficient Solder",
]

UNSUPPORTED_REQUESTED_CLASSES = [
    "Solder Bridges",
    "Misaligned Components",
]


def display_label(raw_label: str) -> str:
    return RAW_TO_DISPLAY.get(raw_label, raw_label.replace("_", " ").title())