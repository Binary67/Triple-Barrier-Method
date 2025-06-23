import numpy as np
import pandas as pd
if not hasattr(np, "NaN"):
    # Older pandas_ta expects numpy.NaN which was removed in numpy>=2.0
    np.NaN = np.nan  # type: ignore[attr-defined]
import pandas_ta as ta
from typing import Any, Callable, Dict, List


class TechnicalIndicator:
    """Compute technical indicators for trading data."""

    def __init__(self, Data: pd.DataFrame, Params: Dict[str, Any]) -> None:
        self.Data = Data.copy()
        self.Params = Params
        self.Methods: Dict[str, Callable[[], pd.DataFrame]] = {}
        self.RegisterIndicator("MACD", self._AddMacd)
        self.RegisterIndicator("EMA", self._AddEma)
        self.RegisterIndicator("SMA", self._AddSma)
        self.RegisterIndicator("RSI", self._AddRsi)
        self.RegisterIndicator("BB", self._AddBb)
        self.RegisterIndicator("MFI", self._AddMfi)

    def RegisterIndicator(self, Name: str, Func: Callable[[], pd.DataFrame]) -> None:
        """Register a new indicator computation method."""
        self.Methods[Name] = Func

    def Apply(self, Name: str) -> pd.DataFrame:
        """Apply a single registered indicator and return resulting DataFrame."""
        if Name not in self.Methods:
            raise ValueError(f"Indicator {Name} not registered")
        return self.Methods[Name]()

    def ApplyAll(self) -> pd.DataFrame:
        """Apply all registered indicators sequentially."""
        for Name in self.Methods:
            self.Methods[Name]()
        return self.Data

    def _AddEma(self) -> pd.DataFrame:
        Windows: List[int] = [int(W) for W in self.Params.get("EMAWindows", [])]
        for Window in Windows:
            Series = ta.ema(self.Data["Close"], length=Window)
            self.Data[Series.name] = Series
        return self.Data

    def _AddSma(self) -> pd.DataFrame:
        Windows: List[int] = [int(W) for W in self.Params.get("SMAWindows", [])]
        for Window in Windows:
            Series = ta.sma(self.Data["Close"], length=Window)
            self.Data[Series.name] = Series
        return self.Data

    def _AddRsi(self) -> pd.DataFrame:
        Windows: List[int] = [int(W) for W in self.Params.get("RSIWindows", [])]
        for Window in Windows:
            Series = ta.rsi(self.Data["Close"], length=Window)
            self.Data[Series.name] = Series
        return self.Data

    def _AddBb(self) -> pd.DataFrame:
        ParamsList = self.Params.get("BBParams", [])
        for Item in ParamsList:
            Window = int(Item.get("Window", 20))
            Std = float(Item.get("Std", 2))
            Bb = ta.bbands(self.Data["Close"], length=Window, std=Std)
            for Column in Bb.columns:
                self.Data[Column] = Bb[Column]
        return self.Data

    def _AddMfi(self) -> pd.DataFrame:
        Windows: List[int] = [int(W) for W in self.Params.get("MFIWindows", [])]
        Required = {"High", "Low", "Close", "Volume"}
        if not Required.issubset(self.Data.columns):
            Missing = Required.difference(self.Data.columns)
            raise ValueError(f"Missing columns required for MFI: {Missing}")
        for Window in Windows:
            Series = ta.mfi(
                high=self.Data["High"],
                low=self.Data["Low"],
                close=self.Data["Close"],
                volume=self.Data["Volume"],
                length=Window,
            )
            self.Data[Series.name] = Series
        return self.Data

    def _AddMacd(self) -> pd.DataFrame:
        ParamsList = self.Params.get("MACDParams", [])
        for Item in ParamsList:
            Fast = int(Item.get("Fast", 12))
            Slow = int(Item.get("Slow", 26))
            Signal = int(Item.get("Signal", 9))
            MacdDf = ta.macd(self.Data["Close"], fast=Fast, slow=Slow, signal=Signal)
            for Column in MacdDf.columns:
                self.Data[Column] = MacdDf[Column]
        return self.Data
