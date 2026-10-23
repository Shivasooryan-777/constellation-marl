"""
tests/test_tee_logger.py - Unit test for src/tee_logger.py.
Verifies that console outputs are accurately captured bit-for-bit in an on-disk log file.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tee_logger import TeeLogger


def main():
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "runs"))
    with TeeLogger(log_dir=log_dir, prefix="test_tee") as logger:
        print("[TEST] Line 1: Logging to console and file.")
        print("[TEST] Line 2: Hardware readout verification line.")
        log_path = logger.log_path

    assert os.path.exists(log_path), f"Log file not found at {log_path}"
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "[TEST] Line 1: Logging to console and file." in content, "Missing line 1 in log"
    assert "[TEST] Line 2: Hardware readout verification line." in content, "Missing line 2 in log"
    print(f"[TEST PASSED] Log successfully written and verified at: {log_path}")


if __name__ == "__main__":
    main()
