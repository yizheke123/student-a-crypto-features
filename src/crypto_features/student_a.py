"""Student A features for cryptocurrency performance analysis.

Student: Zheke Yi
Student ID: 25655150

Features:
- DailyReturn
- Return5
- Return20
- Momentum
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Student A engineered features.

    Required columns:
    - token
    - close

    The features are calculated separately for each token.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with DailyReturn, Return5,
        Return20 and Momentum added.
    """

    required_columns = {"token", "close"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    close = df.groupby("token")["close"]

    df["DailyReturn"] = close.pct_change(1)

    df["Return5"] = close.pct_change(5)

    df["Return20"] = close.pct_change(20)

    df["Momentum"] = (
        df["close"] - close.shift(20)
    )

    return df
