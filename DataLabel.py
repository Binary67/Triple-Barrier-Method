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