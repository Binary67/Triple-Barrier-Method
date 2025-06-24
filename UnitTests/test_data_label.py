from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from DataLabel import DataLabel


def test_triple_barrier_labels() -> None:
    Params = {
        "RiskPct": 5,
        "RiskRewardRatio": 2,
        "TimeLimit": 5,
        "ATRLookback": 1,
        "ATRMultiplier": 1.0,
        "Equity": 10000,
    }
    Data = pd.DataFrame(
        {
            "High": [103, 112, 108, 107, 109],
            "Low": [97, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
        }
    )
    Labeler = DataLabel(Params)
    Labeled = Labeler.Apply("TripleBarrier", Data)
    assert list(Labeled["Label"]) == [2, 1, 1, 1, 1]
    assert list(Labeled["Shares"]) == [83, 41, 71, 97, 83]


def test_triple_barrier_multi_ticker() -> None:
    Params = {
        "RiskPct": 5,
        "RiskRewardRatio": 2,
        "TimeLimit": 5,
        "ATRLookback": 1,
        "ATRMultiplier": 1.0,
        "Equity": 10000,
    }
    Data = pd.DataFrame(
        {
            "High": [103, 112, 108, 107, 109] * 2,
            "Low": [97, 100, 101, 102, 103] * 2,
            "Close": [100, 101, 102, 103, 104] * 2,
            "Ticker": ["AAA"] * 5 + ["BBB"] * 5,
        }
    )
    Labeler = DataLabel(Params)
    Labeled = Labeler.Apply("TripleBarrier", Data)
    FirstLabels = list(Labeled[Labeled["Ticker"] == "AAA"]["Label"])
    SecondLabels = list(Labeled[Labeled["Ticker"] == "BBB"]["Label"])
    FirstShares = list(Labeled[Labeled["Ticker"] == "AAA"]["Shares"])
    SecondShares = list(Labeled[Labeled["Ticker"] == "BBB"]["Shares"])
    assert FirstLabels == [2, 1, 1, 1, 1]
    assert SecondLabels == [2, 1, 1, 1, 1]
    assert FirstShares == [83, 41, 71, 97, 83]
    assert SecondShares == [83, 41, 71, 97, 83]

