from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import logging
from sklearn.metrics import classification_report, f1_score
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
        self.Tickers: List[Any] = []
        Clean = Data.dropna(subset=Features + [LabelColumn])
        if "Ticker" in Clean.columns:
            for Ticker, Group in Clean.groupby("Ticker"):
                self._CreateSamples(Group, Ticker)
        else:
            self._CreateSamples(Clean, None)

    def _CreateSamples(self, Data: pd.DataFrame, Ticker: Any | None) -> None:
        Data = Data.sort_index()
        Values = Data[self.Features].astype(float).values
        Labels = Data[self.LabelColumn].values
        Idx = Data.index
        for Start in range(len(Data) - self.SeqLen + 1):
            End = Start + self.SeqLen
            self.Samples.append(Values[Start:End])
            self.Labels.append(int(Labels[End - 1]))
            self.Indices.append(Idx[End - 1])
            self.Tickers.append(Ticker)

    def __len__(self) -> int:
        return len(self.Samples)

    def __getitem__(self, Index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.tensor(self.Samples[Index], dtype=torch.float32)
        Label = self.Labels[Index]
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
        HiddenParam = Params.get("HiddenSize", 50)
        if isinstance(HiddenParam, list):
            self.HiddenSizes = [int(Size) for Size in HiddenParam]
        else:
            self.HiddenSizes = [int(HiddenParam)]
        self.NumLayers = len(self.HiddenSizes)
        self.ModelPath = Params.get("ModelPath", "LSTMModel.pth")
        InputSize = len(self.Features)
        self.LstmLayers = nn.ModuleList()
        for Idx, Size in enumerate(self.HiddenSizes):
            InSize = InputSize if Idx == 0 else self.HiddenSizes[Idx - 1]
            self.LstmLayers.append(
                nn.LSTM(input_size=InSize, hidden_size=Size, batch_first=True)
            )
        self.Classifier = nn.Linear(self.HiddenSizes[-1], 3)
        self.Criterion = nn.CrossEntropyLoss()
        ParamsList = [P for L in self.LstmLayers for P in L.parameters()]
        ParamsList += list(self.Classifier.parameters())
        self.Optimizer = torch.optim.Adam(ParamsList, lr=self.LearningRate)

    def _TrainLoader(self) -> DataLoader:
        DatasetObj = SequenceDataset(
            self.TrainData, self.Features, self.LabelColumn, self.SequenceLength
        )
        return DataLoader(DatasetObj, batch_size=self.BatchSize, shuffle=False)

    def _ValDataset(self) -> SequenceDataset:
        return SequenceDataset(
            self.ValData, self.Features, self.LabelColumn, self.SequenceLength
        )

    def Train(self) -> None:
        Loader = self._TrainLoader()
        for EpochIdx in range(self.Epochs):
            self.LstmLayers.train()
            self.Classifier.train()
            TotalLoss = 0.0
            TotalCorrect = 0
            TotalSamples = 0
            for XBatch, YBatch in Loader:
                self.Optimizer.zero_grad()
                Output = XBatch
                for Layer in self.LstmLayers:
                    Output, (Hidden, _) = Layer(Output)
                Logits = self.Classifier(Hidden.squeeze(0))
                Loss = self.Criterion(Logits, YBatch)
                Loss.backward()
                self.Optimizer.step()
                TotalLoss += Loss.item() * len(YBatch)
                Preds = torch.argmax(Logits, dim=1)
                TotalCorrect += (Preds == YBatch).sum().item()
                TotalSamples += len(YBatch)
            AvgLoss = TotalLoss / TotalSamples
            Accuracy = TotalCorrect / TotalSamples
            logging.info(
                "Epoch %d/%d - Loss: %.4f - Accuracy: %.4f",
                EpochIdx + 1,
                self.Epochs,
                AvgLoss,
                Accuracy,
            )

    def SaveModel(self, PathStr: str | None = None) -> None:
        PathStr = PathStr or self.ModelPath
        State = {
            "LstmLayers": self.LstmLayers.state_dict(),
            "Classifier": self.Classifier.state_dict(),
        }
        torch.save(State, PathStr)

    def LoadModel(self, PathStr: str | None = None) -> None:
        PathStr = PathStr or self.ModelPath
        State = torch.load(PathStr, map_location="cpu")
        self.LstmLayers.load_state_dict(State["LstmLayers"])
        self.Classifier.load_state_dict(State["Classifier"])

    def Evaluate(self) -> Tuple[float, pd.DataFrame]:
        self.LoadModel()
        DatasetObj = self._ValDataset()
        Predictions: List[int] = []
        Labels: List[int] = []
        Idxs: List[Any] = DatasetObj.Indices
        Tickers: List[Any] = DatasetObj.Tickers
        self.LstmLayers.eval()
        self.Classifier.eval()
        with torch.no_grad():
            for X, Y in DataLoader(DatasetObj, batch_size=self.BatchSize):
                Output = X
                for Layer in self.LstmLayers:
                    Output, (Hidden, _) = Layer(Output)
                Logits = self.Classifier(Hidden.squeeze(0))
                PredTensor = torch.argmax(Logits, dim=1)
                Predictions.extend(PredTensor.numpy().tolist())
                Labels.extend(Y.numpy().tolist())
        F1 = f1_score(Labels, Predictions, average="weighted")
        ValCopy = self.ValData.copy()
        if "Ticker" in ValCopy.columns:
            for Ticker, Idx, Pred in zip(Tickers, Idxs, Predictions):
                Mask = (ValCopy["Ticker"] == Ticker) & (ValCopy.index == Idx)
                ValCopy.loc[Mask, "Prediction"] = Pred
        else:
            for Idx, Pred in zip(Idxs, Predictions):
                ValCopy.loc[Idx, "Prediction"] = Pred
        if "Ticker" in ValCopy.columns:
            ResultMap = {
                (Ticker, Idx): (Pred, Label)
                for Ticker, Idx, Pred, Label in zip(
                    Tickers, Idxs, Predictions, Labels
                )
            }
            AllLabels = sorted(
                ValCopy[self.LabelColumn].dropna().astype(int).unique().tolist()
            )
            for Ticker, Group in ValCopy.groupby("Ticker"):
                TrueLabels: List[int] = []
                PredLabels: List[int] = []
                for RowIdx in Group.index:
                    Key = (Ticker, RowIdx)
                    if Key in ResultMap:
                        PredLab, Lab = ResultMap[Key]
                        PredLabels.append(PredLab)
                        TrueLabels.append(Lab)
                if TrueLabels:
                    Report = classification_report(
                        TrueLabels,
                        PredLabels,
                        labels=AllLabels,
                        zero_division=0,
                    )
                    logging.info(
                        "Classification report for %s:\n%s", Ticker, Report
                    )
        return F1, ValCopy
