import numpy as np
import pandas as pd

if not hasattr(np, "NaN"):
    # Older pandas_ta expected numpy.NaN. Assign for backward compatibility.
    np.NaN = np.nan  # type: ignore[attr-defined]
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

    def _EnsureFloatColumn(self, Column: str) -> None:
        """Ensure a DataFrame column exists with float dtype."""
        if Column in self.Data.columns:
            self.Data[Column] = self.Data[Column].astype(float)
        else:
            self.Data[Column] = np.nan

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
            ColName = f"EMA_{Window}"
            self._EnsureFloatColumn(ColName)
            if "Ticker" in self.Data.columns:
                for _, Group in self.Data.groupby("Ticker"):
                    Series = (
                        Group["Close"].astype(float)
                        .ewm(span=Window, adjust=False, min_periods=1)
                        .mean()
                    )
                    Series.name = ColName
                    self.Data.loc[Group.index, ColName] = Series.astype(float)
            else:
                Series = (
                    self.Data["Close"].astype(float)
                    .ewm(span=Window, adjust=False, min_periods=1)
                    .mean()
                )
                Series.name = ColName
                self.Data[ColName] = Series.astype(float)
        return self.Data

    def _AddSma(self) -> pd.DataFrame:
        Windows: List[int] = [int(W) for W in self.Params.get("SMAWindows", [])]
        for Window in Windows:
            ColName = f"SMA_{Window}"
            self._EnsureFloatColumn(ColName)
            if "Ticker" in self.Data.columns:
                for _, Group in self.Data.groupby("Ticker"):
                    Series = (
                        Group["Close"].astype(float)
                        .rolling(window=Window, min_periods=1)
                        .mean()
                    )
                    Series.name = ColName
                    self.Data.loc[Group.index, ColName] = Series.astype(float)
            else:
                Series = (
                    self.Data["Close"].astype(float)
                    .rolling(window=Window, min_periods=1)
                    .mean()
                )
                Series.name = ColName
                self.Data[ColName] = Series.astype(float)
        return self.Data

    def _AddRsi(self) -> pd.DataFrame:
        Windows: List[int] = [int(W) for W in self.Params.get("RSIWindows", [])]
        for Window in Windows:
            ColName = f"RSI_{Window}"
            self._EnsureFloatColumn(ColName)
            if "Ticker" in self.Data.columns:
                for _, Group in self.Data.groupby("Ticker"):
                    Close = Group["Close"].astype(float)
                    Delta = Close.diff().fillna(0.0)
                    Gain = Delta.clip(lower=0)
                    Loss = -Delta.clip(upper=0)
                    AvgGain = Gain.ewm(alpha=1 / Window, adjust=False, min_periods=1).mean()
                    AvgLoss = Loss.ewm(alpha=1 / Window, adjust=False, min_periods=1).mean()
                    Rs = AvgGain / AvgLoss.replace(0, np.nan)
                    Series = 100 - (100 / (1 + Rs))
                    Series = Series.fillna(0.0)
                    Series.name = ColName
                    self.Data.loc[Group.index, ColName] = Series.astype(float)
            else:
                Close = self.Data["Close"].astype(float)
                Delta = Close.diff().fillna(0.0)
                Gain = Delta.clip(lower=0)
                Loss = -Delta.clip(upper=0)
                AvgGain = Gain.ewm(alpha=1 / Window, adjust=False, min_periods=1).mean()
                AvgLoss = Loss.ewm(alpha=1 / Window, adjust=False, min_periods=1).mean()
                Rs = AvgGain / AvgLoss.replace(0, np.nan)
                Series = 100 - (100 / (1 + Rs))
                Series = Series.fillna(0.0)
                Series.name = ColName
                self._EnsureFloatColumn(ColName)
                self.Data[ColName] = Series.astype(float)
        return self.Data

    def _AddBb(self) -> pd.DataFrame:
        ParamsList = self.Params.get("BBParams", [])
        for Item in ParamsList:
            Window = int(Item.get("Window", 20))
            Std = float(Item.get("Std", 2))
            Columns = {
                "BBL": f"BBL_{Window}_{Std}",
                "BBM": f"BBM_{Window}_{Std}",
                "BBU": f"BBU_{Window}_{Std}",
                "BBB": f"BBB_{Window}_{Std}",
                "BBP": f"BBP_{Window}_{Std}",
            }
            for Col in Columns.values():
                self._EnsureFloatColumn(Col)
            if "Ticker" in self.Data.columns:
                for _, Group in self.Data.groupby("Ticker"):
                    Close = Group["Close"].astype(float)
                    Middle = Close.rolling(window=Window, min_periods=1).mean()
                    StdDev = Close.rolling(window=Window, min_periods=1).std(ddof=0)
                    Lower = Middle - StdDev * Std
                    Upper = Middle + StdDev * Std
                    Bandwidth = (Upper - Lower) / Middle.replace(0, np.nan)
                    Bandwidth = Bandwidth.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                    Percent = (Close - Lower) / (Upper - Lower)
                    Percent = Percent.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                    DataDict = {
                        Columns["BBL"]: Lower,
                        Columns["BBM"]: Middle,
                        Columns["BBU"]: Upper,
                        Columns["BBB"]: Bandwidth,
                        Columns["BBP"]: Percent,
                    }
                    for Col, Series in DataDict.items():
                        Series.name = Col
                        self.Data.loc[Group.index, Col] = Series.astype(float)
            else:
                Close = self.Data["Close"].astype(float)
                Middle = Close.rolling(window=Window, min_periods=1).mean()
                StdDev = Close.rolling(window=Window, min_periods=1).std(ddof=0)
                Lower = Middle - StdDev * Std
                Upper = Middle + StdDev * Std
                Bandwidth = (Upper - Lower) / Middle.replace(0, np.nan)
                Bandwidth = Bandwidth.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                Percent = (Close - Lower) / (Upper - Lower)
                Percent = Percent.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                DataDict = {
                    Columns["BBL"]: Lower,
                    Columns["BBM"]: Middle,
                    Columns["BBU"]: Upper,
                    Columns["BBB"]: Bandwidth,
                    Columns["BBP"]: Percent,
                }
                for Col, Series in DataDict.items():
                    Series.name = Col
                    self.Data[Col] = Series.astype(float)
        return self.Data

    def _AddMfi(self) -> pd.DataFrame:
        Windows: List[int] = [int(W) for W in self.Params.get("MFIWindows", [])]
        Required = {"High", "Low", "Close", "Volume"}
        if not Required.issubset(self.Data.columns):
            Missing = Required.difference(self.Data.columns)
            raise ValueError(f"Missing columns required for MFI: {Missing}")
        for Window in Windows:
            ColName = f"MFI_{Window}"
            self._EnsureFloatColumn(ColName)
            if "Ticker" in self.Data.columns:
                for _, Group in self.Data.groupby("Ticker"):
                    High = Group["High"].astype(float)
                    Low = Group["Low"].astype(float)
                    Close = Group["Close"].astype(float)
                    Volume = Group["Volume"].astype(float)
                    TypicalPrice = (High + Low + Close) / 3.0
                    RawMoneyFlow = TypicalPrice * Volume
                    PositiveFlow = RawMoneyFlow.where(TypicalPrice.diff() > 0, 0.0)
                    NegativeFlow = RawMoneyFlow.where(TypicalPrice.diff() < 0, 0.0)
                    SumPos = PositiveFlow.rolling(window=Window, min_periods=1).sum()
                    SumNeg = NegativeFlow.rolling(window=Window, min_periods=1).sum()
                    Mfi = 100 * SumPos / (SumPos + SumNeg)
                    Mfi = Mfi.fillna(0.0)
                    Mfi.name = ColName
                    self.Data.loc[Group.index, ColName] = Mfi.astype(float)
            else:
                High = self.Data["High"].astype(float)
                Low = self.Data["Low"].astype(float)
                Close = self.Data["Close"].astype(float)
                Volume = self.Data["Volume"].astype(float)
                TypicalPrice = (High + Low + Close) / 3.0
                RawMoneyFlow = TypicalPrice * Volume
                PositiveFlow = RawMoneyFlow.where(TypicalPrice.diff() > 0, 0.0)
                NegativeFlow = RawMoneyFlow.where(TypicalPrice.diff() < 0, 0.0)
                SumPos = PositiveFlow.rolling(window=Window, min_periods=1).sum()
                SumNeg = NegativeFlow.rolling(window=Window, min_periods=1).sum()
                Mfi = 100 * SumPos / (SumPos + SumNeg)
                Mfi = Mfi.fillna(0.0)
                Mfi.name = ColName
                self.Data.loc[:, ColName] = Mfi.astype(float)
        return self.Data

    def _AddMacd(self) -> pd.DataFrame:
        ParamsList = self.Params.get("MACDParams", [])
        for Item in ParamsList:
            Fast = int(Item.get("Fast", 12))
            Slow = int(Item.get("Slow", 26))
            Signal = int(Item.get("Signal", 9))
            ColMacd = f"MACD_{Fast}_{Slow}_{Signal}"
            ColHist = f"MACDh_{Fast}_{Slow}_{Signal}"
            ColSignal = f"MACDs_{Fast}_{Slow}_{Signal}"
            self._EnsureFloatColumn(ColMacd)
            self._EnsureFloatColumn(ColHist)
            self._EnsureFloatColumn(ColSignal)
            if "Ticker" in self.Data.columns:
                for _, Group in self.Data.groupby("Ticker"):
                    Close = Group["Close"].astype(float)
                    EmaFast = Close.ewm(span=Fast, adjust=False, min_periods=1).mean()
                    EmaSlow = Close.ewm(span=Slow, adjust=False, min_periods=1).mean()
                    MacdLine = EmaFast - EmaSlow
                    SignalLine = MacdLine.ewm(span=Signal, adjust=False, min_periods=1).mean()
                    Hist = MacdLine - SignalLine
                    self.Data.loc[Group.index, ColMacd] = MacdLine.astype(float)
                    self.Data.loc[Group.index, ColHist] = Hist.astype(float)
                    self.Data.loc[Group.index, ColSignal] = SignalLine.astype(float)
            else:
                Close = self.Data["Close"].astype(float)
                EmaFast = Close.ewm(span=Fast, adjust=False, min_periods=1).mean()
                EmaSlow = Close.ewm(span=Slow, adjust=False, min_periods=1).mean()
                MacdLine = EmaFast - EmaSlow
                SignalLine = MacdLine.ewm(span=Signal, adjust=False, min_periods=1).mean()
                Hist = MacdLine - SignalLine
                self.Data[ColMacd] = MacdLine.astype(float)
                self.Data[ColHist] = Hist.astype(float)
                self.Data[ColSignal] = SignalLine.astype(float)
        return self.Data
