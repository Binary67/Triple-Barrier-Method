import logging

from LogManager import SetupLogging
from ConfigManager import ConfigManager
from DataDownloader import YFinanceDownloader
from DataLabel import DataLabel
from TechnicalIndicator import TechnicalIndicator
from LSTMModel import LSTMModel
from DataSplitUtils import SplitByDate


def main() -> None:
    SetupLogging()
    Manager = ConfigManager("Params.yaml")
    Params = Manager.LoadParams()
    logging.info("Loaded parameters: %s", Params)

    Downloader = YFinanceDownloader(
        Params.get("Tickers", ["AAPL"]),
        Params.get("StartDate", "2023-01-01"),
        Params.get("EndDate", "2023-01-10"),
        Params.get("Interval", "1d"),
        Params.get("CacheDir", "DataCache"),
    )

    Data = Downloader.DownloadData()
    logging.info("Downloaded %d rows", len(Data))
    if "Ticker" in Data.columns:
        logging.info("Tickers present: %s", Data["Ticker"].unique().tolist())

    Indicators = TechnicalIndicator(Data, Params)
    Data = Indicators.ApplyAll()
    logging.info("Data Columns: %s", Data.columns.tolist())

    Labeler = DataLabel(Params)
    Labeled = Labeler.Apply("TripleBarrier", Data)
    logging.info("Label counts: %s", Labeled["Label"].value_counts().to_dict())

    LstmParams = Params.get("LSTMParams", {})
    Features = LstmParams.get("Features", ["Close"])
    LabelColumn = LstmParams.get("LabelColumn", "Label")

    TrainDf, ValDf = SplitByDate(
        Labeled,
        "2020-01-01",
        "2023-12-31",
        "2024-01-01",
        "2024-12-31",
    )
    Model = LSTMModel(TrainDf, ValDf, Features, LabelColumn, LstmParams)
    Model.Train()
    Model.SaveModel(LstmParams.get("ModelPath"))
    F1, PredDf = Model.Evaluate()
    logging.info("Validation F1: %.4f", F1)
    logging.info("IsNaN Counts: %s", PredDf["Prediction"].isna().sum())
    logging.info("Prediction Value Counts: %s", PredDf["Prediction"].value_counts().to_dict())


if __name__ == "__main__":
    main()
