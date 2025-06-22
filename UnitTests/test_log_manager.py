import os
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from LogManager import SetupLogging


def test_log_creation_and_cleanup(tmp_path: Path) -> None:
    OriginalCwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        LogsDir = tmp_path / "Logs"
        for Index in range(6):
            logging.getLogger().handlers.clear()
            LogFile = SetupLogging()
            logging.getLogger(__name__).info("Log entry %d", Index)
            logging.shutdown()
            assert LogFile.exists()
        LogFiles = sorted(LogsDir.glob("*.log"))
        assert len(LogFiles) == 5
        with LogFiles[-1].open("r", encoding="utf-8") as FH:
            Content = FH.read()
        assert "test_log_manager" in Content
        assert "test_log_creation_and_cleanup" in Content
    finally:
        os.chdir(OriginalCwd)
