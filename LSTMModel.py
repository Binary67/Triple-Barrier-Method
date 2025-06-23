from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    """Dataset for sequential LSTM training."""

    def __init__(
        self, Data: pd.DataFrame, Features: List[str], LabelColumn: str, SeqLen: int
    ) -> None:
        self.Features = Features
        self.LabelColumn = LabelColumn
        self.SeqLen = SeqLen
        self.Samples: List[np.ndarray] = []
        self.Labels: List[int] = []
        self.Indices: List[Any] = []
        Clean = Data.dropna(subset=Features + [LabelColumn])
        if "Ticker" in Clean.columns:
            for _, Group in Clean.groupby("Ticker"):
                self._CreateSamples(Group)
        else:
            self._CreateSamples(Clean)

    def _CreateSamples(self, Data: pd.DataFrame) -> None:
        Data = Data.sort_index()
        Values = Data[self.Features].astype(float).values
        Labels = Data[self.LabelColumn].values
        Idx = Data.index
        for Start in range(len(Data) - self.SeqLen + 1):
            End = Start + self.SeqLen
            self.Samples.append(Values[Start:End])
            self.Labels.append(int(Labels[End - 1]))
            self.Indices.append(Idx[End - 1])

    def __len__(self) -> int:
        return len(self.Samples)

    def __getitem__(self, Index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.tensor(self.Samples[Index], dtype=torch.float32)
        Label = self.Labels[Index] + 1  # map -1,0,1 -> 0,1,2
        Y = torch.tensor(Label, dtype=torch.long)
        return X, Y


class LSTMModel:
    """LSTM model wrapper for training and evaluation."""

    def __init__(
        self,
        TrainData: pd.DataFrame,
        ValData: pd.DataFrame,
        Features: List[str],
        LabelColumn: str,
        Params: Dict[str, Any],
    ) -> None:
        self.TrainData = TrainData
        self.ValData = ValData
        self.Features = Features
        self.LabelColumn = LabelColumn
        self.BatchSize = int(Params.get("BatchSize", 32))
        self.LearningRate = float(Params.get("LearningRate", 0.001))
        self.Epochs = int(Params.get("Epochs", 1))
        self.SequenceLength = int(Params.get("SequenceLength", 5))
        self.HiddenSize = int(Params.get("HiddenSize", 50))
        self.NumLayers = int(Params.get("NumLstmLayers", 1))
        self.ModelPath = Params.get("ModelPath", "LSTMModel.pth")
        InputSize = len(self.Features)
        self.Model = nn.LSTM(
            input_size=InputSize,
            hidden_size=self.HiddenSize,
            num_layers=self.NumLayers,
            batch_first=True,
        )
        self.Classifier = nn.Linear(self.HiddenSize, 3)
        self.Criterion = nn.CrossEntropyLoss()
        self.Optimizer = torch.optim.Adam(
            list(self.Model.parameters()) + list(self.Classifier.parameters()),
            lr=self.LearningRate,
        )

    def _TrainLoader(self) -> DataLoader:
        DatasetObj = SequenceDataset(
            self.TrainData, self.Features, self.LabelColumn, self.SequenceLength
        )
        return DataLoader(DatasetObj, batch_size=self.BatchSize, shuffle=True)

    def _ValDataset(self) -> SequenceDataset:
        return SequenceDataset(
            self.ValData, self.Features, self.LabelColumn, self.SequenceLength
        )

    def Train(self) -> None:
        Loader = self._TrainLoader()
        for _ in range(self.Epochs):
            self.Model.train()
            for XBatch, YBatch in Loader:
                self.Optimizer.zero_grad()
                Output, (Hidden, _) = self.Model(XBatch)
                Logits = self.Classifier(Hidden[-1])
                Loss = self.Criterion(Logits, YBatch)
                Loss.backward()
                self.Optimizer.step()

    def SaveModel(self, PathStr: str | None = None) -> None:
        PathStr = PathStr or self.ModelPath
        State = {
            "Model": self.Model.state_dict(),
            "Classifier": self.Classifier.state_dict(),
        }
        torch.save(State, PathStr)

    def LoadModel(self, PathStr: str | None = None) -> None:
        PathStr = PathStr or self.ModelPath
        State = torch.load(PathStr, map_location="cpu")
        self.Model.load_state_dict(State["Model"])
        self.Classifier.load_state_dict(State["Classifier"])

    def Evaluate(self) -> Tuple[float, pd.DataFrame]:
        DatasetObj = self._ValDataset()
        Predictions: List[int] = []
        Labels: List[int] = []
        Idxs: List[Any] = DatasetObj.Indices
        self.Model.eval()
        with torch.no_grad():
            for X, Y in DataLoader(DatasetObj, batch_size=self.BatchSize):
                Output, (Hidden, _) = self.Model(X)
                Logits = self.Classifier(Hidden[-1])
                PredTensor = torch.argmax(Logits, dim=1)
                Predictions.extend(PredTensor.numpy().tolist())
                Labels.extend(Y.numpy().tolist())
        F1 = f1_score(Labels, Predictions, average="weighted")
        ValCopy = self.ValData.copy()
        for Idx, Pred in zip(Idxs, Predictions):
            ValCopy.loc[Idx, "Prediction"] = Pred - 1
        return F1, ValCopy
