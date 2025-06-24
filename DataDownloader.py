from typing import List, Optional

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

class YFinanceDownloader:
    def __init__(self, Tickers: List[str], StartDate: str, EndDate: str, Interval: str, CacheDir: str = "DataCache") -> None:
        self.Tickers = Tickers
        self.StartDate = pd.to_datetime(StartDate)
        self.EndDate = pd.to_datetime(EndDate)
        self.Interval = Interval
        self.CacheDir = Path(CacheDir)
        self.CacheDir.mkdir(parents=True, exist_ok=True)

    def _GetCachePath(self, Ticker: str) -> Path:
        FileName = (
            f"{Ticker}_{self.StartDate.strftime('%Y%m%d')}_"
            f"{self.EndDate.strftime('%Y%m%d')}_{self.Interval}.pkl"
        )
        return self.CacheDir / FileName

    def _LoadFromCache(self, Ticker: str) -> Optional[pd.DataFrame]:
        PathToFile = self._GetCachePath(Ticker)
        if PathToFile.exists():
            logging.getLogger(__name__).info("Loading %s from cache", Ticker)
            return pd.read_pickle(PathToFile)
        return None

    def _SaveToCache(self, Ticker: str, Data: pd.DataFrame) -> None:
        PathToFile = self._GetCachePath(Ticker)
        try:
            Data.to_pickle(PathToFile)
        except Exception as Err:  # pragma: no cover
            logging.getLogger(__name__).warning("Failed to cache %s: %s", Ticker, Err)

    def _DownloadSingle(self, Ticker: str) -> pd.DataFrame:
        """Download data for a single ticker."""
        Cached = self._LoadFromCache(Ticker)
        if Cached is not None:
            Cached["Ticker"] = Ticker
            return Cached

        HourlyIntervals = ["60m", "1h", "hourly"]
        if self.Interval in HourlyIntervals and (self.EndDate - self.StartDate).days > 14:
            DataFrames = []
            CurrentStart = self.StartDate
            while CurrentStart < self.EndDate:
                CurrentEnd = CurrentStart + pd.Timedelta(days=14)
                if CurrentEnd > self.EndDate:
                    CurrentEnd = self.EndDate
                TempData = yf.download(
                    Ticker,
                    start=CurrentStart.strftime("%Y-%m-%d"),
                    end=CurrentEnd.strftime("%Y-%m-%d"),
                    interval=self.Interval,
                    progress=False,
                )
                if isinstance(TempData.columns, pd.MultiIndex):
                    TempData.columns = TempData.columns.droplevel(1)
                DataFrames.append(TempData)
                CurrentStart = CurrentEnd
            FinalDf = pd.concat(DataFrames) if DataFrames else pd.DataFrame()
        else:
            FinalDf = yf.download(
                Ticker,
                start=self.StartDate.strftime("%Y-%m-%d"),
                end=self.EndDate.strftime("%Y-%m-%d"),
                interval=self.Interval,
                progress=False,
            )
            if isinstance(FinalDf.columns, pd.MultiIndex):
                FinalDf.columns = FinalDf.columns.droplevel(1)

        FinalDf.columns.name = None
        FinalDf["Ticker"] = Ticker
        self._SaveToCache(Ticker, FinalDf)
        return FinalDf

    def DownloadData(self):
        DataFrames = [self._DownloadSingle(T) for T in self.Tickers]
        if DataFrames:
            return pd.concat(DataFrames)
        return pd.DataFrame()
