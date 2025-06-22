import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import yaml  # type: ignore

from ConfigManager import ConfigManager


def test_load_and_save_params(tmp_path: Path) -> None:
    File = tmp_path / "config.yaml"
    Manager = ConfigManager(str(File))
    Manager.SaveParams({"Key": "Value"})
    Params = Manager.LoadParams()
    assert Params["Key"] == "Value"


def test_overwrite_params(tmp_path: Path) -> None:
    File = tmp_path / "config.yaml"
    Manager = ConfigManager(str(File))
    Manager.SaveParams({"A": 1})
    Manager.SaveParams({"A": 2, "B": 3})
    with File.open("r", encoding="utf-8") as FH:
        Data = yaml.safe_load(FH)
    assert Data == {"A": 2, "B": 3}
