"""
Quick hardware readout for S3.3 post-run reporting.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ippo_train import get_system_memory, get_process_memory_mb

tot_gb, used_gb, pct = get_system_memory()
proc_mb = get_process_memory_mb()
print(f"System RAM Total: {tot_gb:.2f} GB | Used: {used_gb:.2f} GB ({pct:.1f}%) | Process Resident RAM: {proc_mb:.2f} MB")
