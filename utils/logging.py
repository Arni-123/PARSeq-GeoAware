# utils/logging.py
"""
Simple logger with timestamp + optional file output
"""

import sys
import time
from pathlib import Path


class Logger:
    def __init__(self, log_file: str = None):
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(self.log_file, "a", encoding="utf-8")
        else:
            self.f = None

    def info(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def get_logger(log_file: str = None):
    return Logger(log_file)