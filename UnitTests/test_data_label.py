from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from DataLabel import DataLabel


def test_triple_barrier_labels() -> None:
    Params = {"RiskPct": 5, "RiskRewardRatio": 2, "TimeLimit": 5}
    Data = pd.DataFrame(
        {
            "High": [105, 111, 108, 107, 109],
            "Low": [98, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
        }
    )
    Labeler = DataLabel(Params)
    Labeled = Labeler.Apply("TripleBarrier", Data)
    assert list(Labeled["Label"]) == [1, 0, 0, 0, 0]


def test_triple_barrier_multi_ticker() -> None:
    Params = {"RiskPct": 5, "RiskRewardRatio": 2, "TimeLimit": 5}
    Data = pd.DataFrame(
        {
            "High": [105, 111, 108, 107, 109] * 2,
            "Low": [98, 100, 101, 102, 103] * 2,
            "Close": [100, 101, 102, 103, 104] * 2,
            "Ticker": ["AAA"] * 5 + ["BBB"] * 5,
        }
    )
    Labeler = DataLabel(Params)
    Labeled = Labeler.Apply("TripleBarrier", Data)
    FirstLabels = list(Labeled[Labeled["Ticker"] == "AAA"]["Label"])
    SecondLabels = list(Labeled[Labeled["Ticker"] == "BBB"]["Label"])
    assert FirstLabels == [1, 0, 0, 0, 0]
    assert SecondLabels == [1, 0, 0, 0, 0]
