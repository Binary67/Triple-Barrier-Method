import logging
from typing import Any, Callable, Dict, Tuple

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

    def CalculateUpDownVolatility(self, Prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Return EWMA up and down volatilities."""
        Span = int(self.Params.get("VolatilitySpan", 20))
        Returns = Prices.pct_change()
        UpVol = Returns.where(Returns > 0).ewm(span=Span, adjust=False).std()
        DownVol = Returns.where(Returns < 0).abs().ewm(span=Span, adjust=False).std()
        UpVol = UpVol.fillna(0)
        DownVol = DownVol.fillna(0)
        return UpVol, DownVol

    def TripleBarrier(self, Data: pd.DataFrame) -> pd.DataFrame:
        """Label data using the Triple Barrier Method."""
        if "Close" not in Data.columns:
            raise ValueError("Data must contain a Close column")
        Data = Data.copy()
        UpVol, DownVol = self.CalculateUpDownVolatility(Data["Close"])
        TpMult = float(self.Params.get("TakeProfitMultiplier", 1.0))
        SlMult = float(self.Params.get("StopLossMultiplier", 1.0))
        Barrier = int(self.Params.get("VerticalBarrier", 5))
        Data["Label"] = 0
        CloseCol = Data.columns.get_loc("Close")
        LabelCol = Data.columns.get_loc("Label")

        for Index in range(len(Data)):
            Price = Data.iat[Index, CloseCol]
            TakeProfit = Price + TpMult * UpVol.iat[Index]
            StopLoss = Price - SlMult * DownVol.iat[Index]
            End = min(Index + Barrier, len(Data) - 1)
            Label = 0
            for J in range(Index + 1, End + 1):
                CurrentPrice = Data.iat[J, CloseCol]
                if CurrentPrice >= TakeProfit:
                    Label = 1
                    break
                if CurrentPrice <= StopLoss:
                    Label = -1
                    break
            Data.iat[Index, LabelCol] = Label
        logging.getLogger(__name__).info("Applied TripleBarrier labeling")
        return Data
