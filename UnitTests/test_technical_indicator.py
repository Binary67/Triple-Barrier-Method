from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from TechnicalIndicator import TechnicalIndicator


def test_indicator_columns_exist() -> None:
    Data = pd.DataFrame(
        {
            "Close": list(range(1, 41)),
            "High": [x + 1 for x in range(1, 41)],
            "Low": [x - 1 for x in range(1, 41)],
            "Volume": list(range(1, 41)),
            "Ticker": ["AAA"] * 20 + ["BBB"] * 20,
        }
    )
    Params = {
        "SMAWindows": [3],
        "EMAWindows": [3],
        "RSIWindows": [3],
        "BBParams": [{"Window": 3, "Std": 2}],
        "MFIWindows": [3],
        "MACDParams": [{"Fast": 3, "Slow": 5, "Signal": 2}],
    }
    Indicator = TechnicalIndicator(Data, Params)
    Result = Indicator.ApplyAll()
    ExpectedCols = {
        "SMA_3",
        "EMA_3",
        "RSI_3",
        "MFI_3",
        "MACD_3_5_2",
        "MACDh_3_5_2",
        "MACDs_3_5_2",
        "BBL_3_2.0",
    }
    assert ExpectedCols.issubset(Result.columns)


def test_mfi_nullable_int() -> None:
    NullableData = pd.DataFrame(
        {
            "Close": pd.Series(range(1, 41), dtype="Int64"),
            "High": pd.Series([x + 1 for x in range(1, 41)], dtype="Int64"),
            "Low": pd.Series([x - 1 for x in range(1, 41)], dtype="Int64"),
            "Volume": pd.Series(range(1, 41), dtype="Int64"),
            "Ticker": ["AAA"] * 20 + ["BBB"] * 20,
        }
    )
    Params = {"MFIWindows": [3]}
    Indicator = TechnicalIndicator(NullableData, Params)
    Result = Indicator.Apply("MFI")
    assert Result["MFI_3"].dtype == float
