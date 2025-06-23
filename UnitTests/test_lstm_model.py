from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from LSTMModel import LSTMModel


def test_lstm_training_and_evaluation() -> None:
    Data = pd.DataFrame(
        {
            "Feature": list(range(10)) + list(range(10, 20)),
            "Label": [-1, -1, 0, 1, 1] * 4,
            "Ticker": ["AAA"] * 10 + ["BBB"] * 10,
        }
    )
    Params = {
        "BatchSize": 2,
        "LearningRate": 0.01,
        "Epochs": 1,
        "SequenceLength": 3,
        "HiddenSize": 4,
        "NumLstmLayers": 1,
    }
    Train = Data.iloc[:12]
    Val = Data.iloc[12:]
    Model = LSTMModel(Train, Val, ["Feature"], "Label", Params)
    Model.Train()
    F1, PredDf = Model.Evaluate()
    assert "Prediction" in PredDf.columns
    assert isinstance(F1, float)
