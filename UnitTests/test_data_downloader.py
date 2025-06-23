from pathlib import Path
from unittest.mock import patch
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from DataDownloader import YFinanceDownloader


def test_multi_ticker_download() -> None:
    def fake_download(ticker: str, start: str, end: str, interval: str, progress: bool) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [1, 2],
                "High": [2, 3],
                "Low": [0, 1],
                "Close": [1, 2],
                "Volume": [10, 10],
            },
            index=pd.date_range(start="2020-01-01", periods=2, freq="D"),
        )

    with patch("yfinance.download", side_effect=fake_download) as download_mock:
        downloader = YFinanceDownloader(["AAA", "BBB"], "2020-01-01", "2020-01-05", "1d")
        df = downloader.DownloadData()
        assert set(df["Ticker"].unique()) == {"AAA", "BBB"}
        assert len(df) == 4
        assert download_mock.call_count == 2

