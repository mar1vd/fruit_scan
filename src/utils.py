"""Small formatting helpers for the FruitScan application."""


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def display_class_name(name: str) -> str:
    return name.replace("_", " ").strip()
