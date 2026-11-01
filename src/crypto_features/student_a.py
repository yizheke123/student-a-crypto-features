"""Student A feature engineering.

Student: Zheke Yi
Student ID: 25655150

Features:
- DailyReturn
- Return5
- Return20
- Momentum
"""

import pandas as pd


REQUIRED_COLUMNS = ["close"]


def _validate_input(df: pd.DataFrame) -> None:
    """Validate the input DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-day percentage return."""

    result = df.copy()

    result["DailyReturn"] = (
        result["close"].pct_change()
    )

    return result


def add_return5(df: pd.DataFrame) -> pd.DataFrame:
    """Add five-day percentage return."""

    result = df.copy()

    result["Return5"] = (
        result["close"].pct_change(5)
    )

    return result


def add_return20(df: pd.DataFrame) -> pd.DataFrame:
    """Add twenty-day percentage return."""

    result = df.copy()

    result["Return20"] = (
        result["close"].pct_change(20)
    )

    return result


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum feature."""

    result = df.copy()

    result["Momentum"] = (
        result["close"]
        - result["close"].shift(20)
    )

    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all Student A engineered features."""

    _validate_input(df)

    result = df.copy()

    # Sort chronologically when a date column is available.
    if "date" in result.columns:
        result = result.sort_values("date")

    result = add_daily_return(result)
    result = add_return5(result)
    result = add_return20(result)
    result = add_momentum(result)

    return result
