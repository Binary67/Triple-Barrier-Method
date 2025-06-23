from typing import List

import pandas as pd
import yfinance as yf

class YFinanceDownloader:
    def __init__(self, Tickers: List[str], StartDate: str, EndDate: str, Interval: str) -> None:
        self.Tickers = Tickers
        self.StartDate = pd.to_datetime(StartDate)
        self.EndDate = pd.to_datetime(EndDate)
        self.Interval = Interval

    def _DownloadSingle(self, Ticker: str) -> pd.DataFrame:
        """Download data for a single ticker."""
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
        return FinalDf

    def DownloadData(self):
        DataFrames = [self._DownloadSingle(T) for T in self.Tickers]
        if DataFrames:
            return pd.concat(DataFrames)
        return pd.DataFrame()
