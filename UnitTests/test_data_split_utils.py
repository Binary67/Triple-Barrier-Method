from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from DataSplitUtils import SplitByDate


def test_split_by_date() -> None:
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"A": range(10)}, index=dates)
    train, val = SplitByDate(df, "2020-01-01", "2020-01-05", "2020-01-06", "2020-01-10")
    assert len(train) == 5
    assert len(val) == 5
    assert list(train.columns) == list(val.columns)
