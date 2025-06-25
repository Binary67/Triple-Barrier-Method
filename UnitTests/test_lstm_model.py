from pathlib import Path
import sys

import os
import logging
from typing import Any

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from LSTMModel import LSTMModel


def test_lstm_training_and_evaluation(caplog: Any) -> None:
    Data = pd.DataFrame(
        {
            "Feature": list(range(10)) + list(range(10, 20)),
            "Label": [0, 0, 1, 2, 2] * 4,
            "Ticker": ["AAA"] * 10 + ["BBB"] * 10,
        }
    )
    Params = {
        "BatchSize": 2,
        "LearningRate": 0.01,
        "Epochs": 1,
        "SequenceLength": 3,
        "HiddenSize": [4],
        "ModelPath": "TempModel.pth",
    }
    Train = Data.iloc[:12]
    Val = Data.iloc[12:]
    Model = LSTMModel(Train, Val, ["Feature"], "Label", Params)
    Model.Train()
    Model.SaveModel()
    caplog.set_level(logging.INFO)
    F1, PredDf = Model.Evaluate()
    os.remove("TempModel.pth")
    assert "Prediction" in PredDf.columns
    assert isinstance(F1, float)
    Reports = [
        Record.getMessage()
        for Record in caplog.records
        if "Classification report" in Record.getMessage()
    ]
    assert len(Reports) == 1
    Text = Reports[0]
    assert " 0" in Text
    assert " 1" in Text
    assert " 2" in Text


def test_classification_report_includes_all_labels(caplog: Any) -> None:
    Data = pd.DataFrame(
        {
            "Feature": list(range(12)),
            "Label": [0, 0, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2],
            "Ticker": ["AAA"] * 6 + ["BBB"] * 6,
        }
    )
    Params = {
        "BatchSize": 2,
        "LearningRate": 0.01,
        "Epochs": 1,
        "SequenceLength": 2,
        "HiddenSize": [4],
        "ModelPath": "TempModel.pth",
    }
    Train = Data.iloc[[0, 1, 2, 3, 6, 7, 8, 9]]
    Val = Data.drop(Train.index)
    Model = LSTMModel(Train, Val, ["Feature"], "Label", Params)
    Model.Train()
    Model.SaveModel()
    caplog.set_level(logging.INFO)
    _, _ = Model.Evaluate()
    os.remove("TempModel.pth")
    Reports = [
        Record.getMessage()
        for Record in caplog.records
        if "Classification report" in Record.getMessage()
    ]
    assert len(Reports) == 1
    Text = Reports[0]
    assert " 0" in Text
    assert " 1" in Text
    assert " 2" in Text


def test_model_device_selection() -> None:
    Data = pd.DataFrame({"Feature": [0, 1, 2, 3], "Label": [0, 1, 0, 1]})
    Params = {
        "BatchSize": 1,
        "LearningRate": 0.01,
        "Epochs": 1,
        "SequenceLength": 2,
        "HiddenSize": [2],
    }
    Model = LSTMModel(Data, Data, ["Feature"], "Label", Params)
    Expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert Model.Device == Expected


def test_class_weights_computation() -> None:
    Data = pd.DataFrame(
        {
            "Feature": list(range(9)),
            "Label": [0] * 4 + [1] * 3 + [2] * 2,
        }
    )
    Params = {
        "BatchSize": 1,
        "LearningRate": 0.01,
        "Epochs": 1,
        "SequenceLength": 2,
        "HiddenSize": [2],
    }
    Model = LSTMModel(Data, Data, ["Feature"], "Label", Params)
    Counts = Data["Label"].value_counts().sort_index()
    NumClasses = len(Counts)
    TotalSamples = len(Data)
    ExpectedWeights = torch.tensor(
        [TotalSamples / (NumClasses * Count) for Count in Counts], dtype=torch.float32
    )
    assert torch.allclose(Model.ClassWeights.cpu(), ExpectedWeights)


def test_dropout_parameter() -> None:
    Data = pd.DataFrame({"Feature": [0, 1, 2, 3], "Label": [0, 1, 0, 1]})
    Params = {
        "BatchSize": 1,
        "LearningRate": 0.01,
        "Epochs": 1,
        "SequenceLength": 2,
        "HiddenSize": [2],
        "DropoutRate": 0.5,
    }
    Model = LSTMModel(Data, Data, ["Feature"], "Label", Params)
    assert isinstance(Model.Dropout, torch.nn.Dropout)
    assert abs(Model.Dropout.p - 0.5) < 1e-6


def test_focal_loss_selection() -> None:
    Data = pd.DataFrame({"Feature": [0, 1, 2, 3], "Label": [0, 1, 0, 1]})
    Params = {
        "BatchSize": 1,
        "LearningRate": 0.01,
        "Epochs": 1,
        "SequenceLength": 2,
        "HiddenSize": [2],
        "LossFunction": "focal",
    }
    Model = LSTMModel(Data, Data, ["Feature"], "Label", Params)
    from LSTMModel import FocalLoss

    assert isinstance(Model.Criterion, FocalLoss)
