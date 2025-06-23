import logging

from LogManager import SetupLogging
from ConfigManager import ConfigManager
from DataDownloader import YFinanceDownloader
from DataLabel import DataLabel
from TechnicalIndicator import TechnicalIndicator
from LSTMModel import LSTMModel


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
    SplitIdx = int(len(Labeled) * 0.8)
    TrainDf = Labeled.iloc[:SplitIdx]
    ValDf = Labeled.iloc[SplitIdx:]
    Model = LSTMModel(TrainDf, ValDf, Features, LabelColumn, LstmParams)
    Model.Train()
    F1, PredDf = Model.Evaluate()
    logging.info("Validation F1: %.4f", F1)
    Model.SaveModel(LstmParams.get("ModelPath"))


if __name__ == "__main__":
    main()
