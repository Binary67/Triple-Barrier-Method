import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from DataLabel import DataLabel


def test_calculate_up_down_volatility() -> None:
    Prices = pd.Series([100, 102, 101, 103, 104], dtype=float)
    Params = {"VolatilitySpan": 2}
    Labeler = DataLabel(Params)
    UpVol, DownVol = Labeler.CalculateUpDownVolatility(Prices)
    assert len(UpVol) == len(Prices)
    assert (UpVol >= 0).all()
    assert (DownVol >= 0).all()


def test_triple_barrier_label() -> None:
    Prices = pd.Series([100, 101, 102, 103, 104, 105], dtype=float)
    Data = pd.DataFrame({"Close": Prices})
    Params = {
        "VolatilitySpan": 2,
        "TakeProfitMultiplier": 1.0,
        "StopLossMultiplier": 1.0,
        "VerticalBarrier": 3,
    }
    Labeler = DataLabel(Params)
    Result = Labeler.Apply("TripleBarrier", Data)
    assert "Label" in Result.columns
    # With strictly increasing prices take profit should trigger
    assert Result["Label"].iloc[0] == 1
