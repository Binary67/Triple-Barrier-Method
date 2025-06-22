import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ConfigManager import ConfigManager
from DataDownloader import YFinanceDownloader


def SetupLogging() -> None:
    LogsDir = Path("Logs")
    LogsDir.mkdir(exist_ok=True)
    Handler = RotatingFileHandler(
        LogsDir / "app.log",
        maxBytes=1024 * 1024,
        backupCount=3,
    )
    LoggingFormat = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, handlers=[Handler], format=LoggingFormat)


def main() -> None:
    SetupLogging()
    Manager = ConfigManager("Params.yaml")
    Params = Manager.LoadParams()
    logging.info("Loaded parameters: %s", Params)

    Downloader = YFinanceDownloader(
        Params.get("Ticker", "AAPL"),
        Params.get("StartDate", "2023-01-01"),
        Params.get("EndDate", "2023-01-10"),
        Params.get("Interval", "1d"),
    )

    Data = Downloader.DownloadData()
    logging.info("Downloaded %d rows", len(Data))

    Manager.SaveParams({"LastDownloadedRows": int(len(Data))})
    logging.info("Updated parameters saved to Params.yaml")


if __name__ == "__main__":
    main()
