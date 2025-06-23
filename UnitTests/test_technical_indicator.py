from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from TechnicalIndicator import TechnicalIndicator


def test_indicator_columns_exist() -> None:
    Data = pd.DataFrame(
        {
            "Close": list(range(1, 40)),
            "High": [x + 1 for x in range(1, 40)],
            "Low": [x - 1 for x in range(1, 40)],
            "Volume": list(range(1, 40)),
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
