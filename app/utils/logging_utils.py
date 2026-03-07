import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler

# ---- IST timezone (UTC+5:30) ----
IST = timezone(timedelta(hours=5, minutes=30))


# ---- ANSI color codes for terminal output ----
class _Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    DEBUG = "\033[36m"  # Cyan
    INFO = ""  # Default
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[35m"  # Magenta

    TIMESTAMP = "\033[90m"  # Dark gray
    NAME = "\033[96m"  # Light cyan


# ---- Human-readable colored formatter for terminal ----
class ISTColorFormatter(logging.Formatter):
    """
    Terminal formatter:
      - Timestamps in IST (UTC+5:30)
      - ANSI colors per log level
      - Shows filename and line number
    """

    LEVEL_COLORS = {
        logging.DEBUG: _Colors.DEBUG,
        logging.INFO: _Colors.INFO,
        logging.WARNING: _Colors.WARNING,
        logging.ERROR: _Colors.ERROR,
        logging.CRITICAL: _Colors.CRITICAL,
    }

    LEVEL_LABELS = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def formatTime(self, record: logging.LogRecord, datefmt=None) -> str:  # noqa: N802
        ist_dt = datetime.fromtimestamp(record.created, tz=IST)
        return ist_dt.strftime("%Y-%m-%d %H:%M:%S IST")

    def format(self, record: logging.LogRecord) -> str:
        level_color = self.LEVEL_COLORS.get(record.levelno, _Colors.RESET)
        label = self.LEVEL_LABELS.get(record.levelno, record.levelname)

        timestamp = self.formatTime(record)
        filename = record.filename
        lineno = record.lineno
        logger_name = record.name

        if self.use_color:
            ts_part = f"{_Colors.TIMESTAMP}{timestamp}{_Colors.RESET}"
            level_part = f"{_Colors.BOLD}{level_color}{label:<8}{_Colors.RESET}"
            file_part = f"{_Colors.DIM}{filename}:{lineno}{_Colors.RESET}"
            name_part = f"{_Colors.NAME}{logger_name}{_Colors.RESET}"
            msg_part = f"{level_color}{record.getMessage()}{_Colors.RESET}"
        else:
            ts_part = timestamp
            level_part = label.ljust(8)
            file_part = f"{filename}:{lineno}"
            name_part = logger_name
            msg_part = record.getMessage()

        formatted = f"[{ts_part}] [{level_part}] [{file_part}] {name_part} - {msg_part}"

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            formatted += "\n" + self.formatStack(record.stack_info)

        return formatted


# ---- Structured JSON formatter for log file (Grafana-ready) ----
class JSONFormatter(logging.Formatter):
    """
    File formatter: one JSON object per line (NDJSON).
    Compatible with Grafana Loki and similar log aggregators.

    Fields: timestamp, level, logger, file, line, function, message,
            and any extra fields passed via extra={} on the log call.
    """

    # Fields that are part of the standard LogRecord — not re-emitted as extras
    _STANDARD_KEYS = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "taskName",
        }
    )

    def formatTime(self, record: logging.LogRecord, datefmt=None) -> str:  # noqa: N802
        ist_dt = datetime.fromtimestamp(record.created, tz=IST)
        return ist_dt.isoformat()

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)

        # Attach JSON-serialisable extras (e.g. extra={"request_id": "abc"})
        for key, value in record.__dict__.items():
            if key not in self._STANDARD_KEYS:
                try:
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        return json.dumps(log_entry, ensure_ascii=False)


# ---- Root logger bootstrap (runs once) ----
_root_bootstrapped = False


def _bootstrap_root_logger() -> None:
    global _root_bootstrapped
    if _root_bootstrapped:
        return
    _root_bootstrapped = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # --- stdout: colored human-readable, INFO+ ---
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(ISTColorFormatter(use_color=True))
    root.addHandler(stdout_handler)

    # --- file: rotating JSON (NDJSON), DEBUG+ ---
    log_dir = os.environ.get("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "app.log")

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=50 * 1024 * 1024,  # 50 MB per file
        backupCount=5,  # keep app.log.1 … app.log.5
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)

    # --- suppress noisy third-party loggers ---
    for name, level in {
        "celery": logging.INFO,
        "celery.task": logging.DEBUG,
        "celery.worker": logging.INFO,
        "uvicorn": logging.INFO,
        "uvicorn.error": logging.INFO,
        "uvicorn.access": logging.INFO,
        "gunicorn": logging.INFO,
        "redis": logging.WARNING,
        "kombu": logging.WARNING,
        "weasyprint": logging.CRITICAL,
        "weasyprint.progress": logging.CRITICAL,
        "fontTools": logging.CRITICAL,
    }.items():
        logging.getLogger(name).setLevel(level)

    _log_startup_banner(log_file_path)


def _log_startup_banner(log_file_path: str) -> None:
    log = logging.getLogger("Scron")
    environment = os.environ.get("ENVIRONMENT", "unknown")
    log.info("=" * 80)
    log.info("🚀 Logging initialized for: scron")
    log.info(f"   Environment : {environment}")
    log.info(f"   Process ID  : {os.getpid()}")
    log.info(f"   Log file    : {os.path.abspath(log_file_path)}")
    log.info("   stdout      : colored human-readable (INFO+)")
    log.info("   file        : NDJSON / Grafana-ready (DEBUG+)")
    log.info("   Timezone    : IST (UTC+5:30)")
    log.info("=" * 80)


# ---- Public API ----


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger. Bootstraps root logger on first call.

    Usage:
        log = get_logger(__name__)
        log.info("hello")
        log.warning("something off", extra={"order_id": 42})
    """
    _bootstrap_root_logger()
    return logging.getLogger(name)


# ---- Structured helper functions ----


def log_task_start(task_name: str, **kwargs) -> None:
    log = logging.getLogger("Digitization")
    log.info("=" * 80)
    log.info(f"🚀 TASK STARTED: {task_name}")
    for key, value in kwargs.items():
        log.info(f"   {key}: {value}")
    log.info("=" * 80)


def log_task_end(task_name: str, duration: float = None, **kwargs) -> None:
    log = logging.getLogger("Digitization")
    log.info("=" * 80)
    log.info(f"✅ TASK COMPLETED: {task_name}")
    if duration is not None:
        log.info(f"   Duration: {duration:.2f}s")
    for key, value in kwargs.items():
        log.info(f"   {key}: {value}")
    log.info("=" * 80)


def log_task_error(task_name: str, error: Exception, **kwargs) -> None:
    log = logging.getLogger("Digitization")
    log.error("=" * 80)
    log.error(f"❌ TASK FAILED: {task_name}")
    log.error(f"   Error: {type(error).__name__}: {str(error)}")
    for key, value in kwargs.items():
        log.error(f"   {key}: {value}")
    log.error("=" * 80)
    log.error(f"Full Traceback:\n{traceback.format_exc()}")
    log.error("=" * 80)


# ---- Context manager for timed operations ----


class LogTimer:
    """Context manager that logs the duration of a block."""

    def __init__(self, operation_name: str, log_level: int = logging.INFO):
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time: datetime | None = None
        self._log = logging.getLogger("Digitization")

    def __enter__(self):
        self.start_time = datetime.now()
        self._log.log(self.log_level, f"⏱️  Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self._log.log(
                self.log_level,
                f"✅ Completed: {self.operation_name} ({duration:.2f}s)",
            )
        else:
            self._log.error(
                f"❌ Failed: {self.operation_name} ({duration:.2f}s) "
                f"- {exc_type.__name__}: {exc_val}"
            )
        return False  # don't suppress exceptions
