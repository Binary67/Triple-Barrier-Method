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
