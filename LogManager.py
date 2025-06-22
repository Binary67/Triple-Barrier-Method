import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


def _CleanupOldLogs(LogsDir: Path) -> None:
    LogFiles = sorted(LogsDir.glob("*.log"), key=lambda P: P.stat().st_mtime)
    while len(LogFiles) > 5:
        OldLog = LogFiles.pop(0)
        try:
            OldLog.unlink()
        except OSError:
            logging.getLogger(__name__).warning("Could not delete old log %s", OldLog)


def SetupLogging() -> Path:
    LogsDir = Path("Logs")
    LogsDir.mkdir(exist_ok=True)
    LogFile = LogsDir / f"{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    Handler = RotatingFileHandler(LogFile, maxBytes=1024 * 1024, backupCount=1)
    LoggingFormat = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
    logging.basicConfig(level=logging.INFO, handlers=[Handler], format=LoggingFormat)
    _CleanupOldLogs(LogsDir)
    return LogFile
