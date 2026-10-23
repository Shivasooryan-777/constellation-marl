"""
src/tee_logger.py - Dual-stream console and file logger for Constellation runs.

Ensures that all console output (stdout & stderr), including hardware baselines,
training iteration progress, and evaluation outputs, is simultaneously mirrored
to a persistent, timestamped log file on disk.

Usage:
    from src.tee_logger import TeeLogger

    with TeeLogger(log_dir="logs/runs", prefix="ippo_sweep") as logger:
        # Everything printed here goes to both console and the log file
        print(...)
    
    # Or manual start/stop:
    logger = TeeLogger(log_dir="logs/runs", prefix="ippo_sweep")
    logger.start()
    ...
    logger.stop()
"""

import os
import sys
import time
from typing import Optional, TextIO


class DualStream(object):
    """Duplicates writes to both the original stream and a file."""

    def __init__(self, original_stream: TextIO, file_stream: TextIO):
        self.original_stream = original_stream
        self.file_stream = file_stream

    def write(self, data: str) -> int:
        self.original_stream.write(data)
        self.original_stream.flush()
        self.file_stream.write(data)
        self.file_stream.flush()
        return len(data)

    def flush(self) -> None:
        self.original_stream.flush()
        self.file_stream.flush()

    def isatty(self) -> bool:
        return getattr(self.original_stream, "isatty", lambda: False)()


class TeeLogger:
    """Manages redirection of sys.stdout and sys.stderr to a persistent log file."""

    def __init__(self, log_dir: str = "logs/runs", prefix: str = "run"):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{prefix}_{timestamp}.log"
        self.log_path = os.path.join(self.log_dir, self.log_filename)
        self._file: Optional[TextIO] = None
        self._orig_stdout: Optional[TextIO] = None
        self._orig_stderr: Optional[TextIO] = None

    def start(self) -> str:
        """Starts tee logging, returning the path to the log file."""
        self._file = open(self.log_path, mode="w", encoding="utf-8", buffering=1)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = DualStream(self._orig_stdout, self._file)
        sys.stderr = DualStream(self._orig_stderr, self._file)
        return self.log_path

    def stop(self) -> None:
        """Restores original streams and closes log file."""
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
            self._orig_stdout = None
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
            self._orig_stderr = None
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
