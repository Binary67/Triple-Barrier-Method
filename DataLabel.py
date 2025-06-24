from typing import Any, Callable, Dict

import math
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
        AtrLookback = int(self.Params.get("ATRLookback", 14))
        AtrMultiplier = float(self.Params.get("ATRMultiplier", 1.0))
        Equity = float(self.Params.get("Equity", 10000))

        Required = {"High", "Low", "Close"}
        if not Required.issubset(Data.columns):
            Missing = Required.difference(Data.columns)
            raise ValueError(f"Missing columns required for labeling: {Missing}")

        if "Ticker" in Data.columns:
            Groups = []
            for _, Group in Data.groupby("Ticker", group_keys=False):
                Groups.append(
                    self._TripleBarrierSingle(
                        Group,
                        RiskPct,
                        RiskRewardRatio,
                        TimeLimit,
                        AtrLookback,
                        AtrMultiplier,
                        Equity,
                    )
                )
            return pd.concat(Groups).sort_index()
        return self._TripleBarrierSingle(
            Data,
            RiskPct,
            RiskRewardRatio,
            TimeLimit,
            AtrLookback,
            AtrMultiplier,
            Equity,
        )

    def _TripleBarrierSingle(
        self,
        Data: pd.DataFrame,
        RiskPct: float,
        RiskRewardRatio: float,
        TimeLimit: int,
        AtrLookback: int,
        AtrMultiplier: float,
        Equity: float,
    ) -> pd.DataFrame:
        Data = Data.copy()
        HighLow = Data["High"] - Data["Low"]
        HighClose = (Data["High"] - Data["Close"].shift()).abs()
        LowClose = (Data["Low"] - Data["Close"].shift()).abs()
        TrueRange = pd.concat([HighLow, HighClose, LowClose], axis=1).max(axis=1)
        Data["ATR"] = TrueRange.rolling(window=AtrLookback, min_periods=1).mean()

        Labels = []
        SharesList = []
        for StartIdx in range(len(Data)):
            Price = Data.at[Data.index[StartIdx], "Close"]
            AtrValue = Data.at[Data.index[StartIdx], "ATR"]
            StopLoss = Price - AtrMultiplier * AtrValue
            TakeProfit = Price + AtrMultiplier * AtrValue * RiskRewardRatio
            RiskPerShare = Price - StopLoss
            if RiskPerShare <= 0:
                Shares = 0
            else:
                RiskBased = (Equity * RiskPct) / RiskPerShare
                EquityBased = Equity / Price
                Shares = math.floor(min(RiskBased, EquityBased))
            SharesList.append(int(Shares))
            EndIdx = min(StartIdx + TimeLimit, len(Data) - 1)
            Slice = Data.iloc[StartIdx : EndIdx + 1]

            TpMask = Slice["High"] >= TakeProfit
            SlMask = Slice["Low"] <= StopLoss
            TpIdx = Slice.index[TpMask].min() if TpMask.any() else None
            SlIdx = Slice.index[SlMask].min() if SlMask.any() else None

            if TpIdx is not None and SlIdx is not None:
                Label = (
                    2
                    if Slice.index.get_loc(TpIdx) <= Slice.index.get_loc(SlIdx)
                    else 0
                )
            elif TpIdx is not None:
                Label = 2
            elif SlIdx is not None:
                Label = 0
            else:
                Label = 1
            Labels.append(Label)
        Data["Label"] = Labels
        Data["Shares"] = SharesList
        Data = Data.drop(columns=["ATR"])
        return Data
