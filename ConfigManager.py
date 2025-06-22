import logging
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore


class ConfigManager:
    """Handle loading and saving of YAML configuration files."""

    def __init__(self, FilePath: str) -> None:
        self.FilePath = Path(FilePath)
        if not self.FilePath.exists():
            logging.getLogger(__name__).info(
                "Configuration file %s not found. Creating with empty dict.",
                self.FilePath,
            )
            self.SaveParams({})

    def LoadParams(self) -> Dict[str, Any]:
        """Load parameters from YAML file."""
        if not self.FilePath.exists():
            return {}
        with self.FilePath.open("r", encoding="utf-8") as File:
            Data = yaml.safe_load(File) or {}
        return Data

    def SaveParams(self, Params: Dict[str, Any]) -> None:
        """Save parameters to YAML file, overwriting existing entries."""
        Existing = self.LoadParams()
        Existing.update(Params)
        with self.FilePath.open("w", encoding="utf-8") as File:
            yaml.safe_dump(Existing, File)
