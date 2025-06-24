from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from TechnicalIndicator import TechnicalIndicator


def test_per_ticker_z_score() -> None:
    df = pd.DataFrame({
        "Value": [1, 2, 3, 4],
        "Ticker": ["AAA", "AAA", "BBB", "BBB"],
        "Label": [0, 0, 0, 0],
        "Shares": [1, 1, 1, 1],
    })
    result, scalers = TechnicalIndicator.PerTickerZScore(df)
    assert pytest.approx(result.loc[0, "Value"]) == -1.0
    assert pytest.approx(result.loc[1, "Value"]) == 1.0
    assert set(scalers.keys()) == {"AAA", "BBB"}

    new_df = pd.DataFrame({
        "Value": [5, 6],
        "Ticker": ["AAA", "BBB"],
        "Label": [0, 0],
        "Shares": [1, 1],
    })
    transformed, _ = TechnicalIndicator.PerTickerZScore(new_df, scalers)
    assert pytest.approx(transformed.loc[0, "Value"]) == (5 - 1.5) / 0.5
    assert pytest.approx(transformed.loc[1, "Value"]) == (6 - 3.5) / 0.5
