from typing import Any, Callable, Dict

import pandas as pd


class DataLabel:
    """Generic labeling utilities for trading data."""

    def __init__(self, Params: Dict[str, Any]) -> None:
        self.Params = Params
        self.Methods: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}
        self.RegisterMethod("TripleBarrier", self.TripleBarrier)

    def RegisterMethod(self, Name: str, Func: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """Register a new labeling method."""
        self.Methods[Name] = Func

    def Apply(self, Name: str, Data: pd.DataFrame) -> pd.DataFrame:
        """Apply a registered labeling method to Data."""
        if Name not in self.Methods:
            raise ValueError(f"Method {Name} not registered")
        return self.Methods[Name](Data)

    def TripleBarrier(self, Data: pd.DataFrame) -> pd.DataFrame:
        """Label data using the Triple Barrier Method with risk management."""
        RiskPct = float(self.Params.get("RiskPct", 1)) / 100
        RiskRewardRatio = float(self.Params.get("RiskRewardRatio", 1))
        TimeLimit = int(self.Params.get("TimeLimit", 5))

        Required = {"High", "Low", "Close"}
        if not Required.issubset(Data.columns):
            Missing = Required.difference(Data.columns)
            raise ValueError(f"Missing columns required for labeling: {Missing}")

        Data = Data.copy()
        Labels = []
        for StartIdx in range(len(Data)):
            Price = Data.at[Data.index[StartIdx], "Close"]
            StopLoss = Price * (1 - RiskPct)
            TakeProfit = Price * (1 + RiskPct * RiskRewardRatio)
            EndIdx = min(StartIdx + TimeLimit, len(Data) - 1)
            Slice = Data.iloc[StartIdx : EndIdx + 1]

            TpMask = Slice["High"] >= TakeProfit
            SlMask = Slice["Low"] <= StopLoss
            TpIdx = Slice.index[TpMask].min() if TpMask.any() else None
            SlIdx = Slice.index[SlMask].min() if SlMask.any() else None

            if TpIdx is not None and SlIdx is not None:
                Label = 1 if Slice.index.get_loc(TpIdx) <= Slice.index.get_loc(SlIdx) else -1
            elif TpIdx is not None:
                Label = 1
            elif SlIdx is not None:
                Label = -1
            else:
                Label = 0
            Labels.append(Label)
        Data["Label"] = Labels
        return Data
