from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ConfigManager import ConfigManager


def test_time_limit_in_params() -> None:
    Manager = ConfigManager("Params.yaml")
    Params = Manager.LoadParams()
    assert "TimeLimit" in Params
