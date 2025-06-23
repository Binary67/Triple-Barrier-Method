import pandas as pd
from typing import Tuple


def SplitByDate(
    Data: pd.DataFrame,
    TrainStart: str,
    TrainEnd: str,
    ValStart: str,
    ValEnd: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split Data into training and validation sets based on index dates."""
    Index = pd.to_datetime(Data.index)
    TrainMask = (Index >= pd.to_datetime(TrainStart)) & (Index <= pd.to_datetime(TrainEnd))
    ValMask = (Index >= pd.to_datetime(ValStart)) & (Index <= pd.to_datetime(ValEnd))
    TrainDf = Data.loc[TrainMask].copy()
    ValDf = Data.loc[ValMask].copy()
    # Ensure consistent columns
    MissingCols = set(TrainDf.columns).symmetric_difference(ValDf.columns)
    if MissingCols:
        ValDf = ValDf.reindex(columns=TrainDf.columns)
    return TrainDf, ValDf
