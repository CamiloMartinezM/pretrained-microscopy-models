"""Configuration module for setting up paths and logging."""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_SEGMENTATION_DATA = PROJ_ROOT / "benchmark_segmentation_data"

# TZINFO
TZINFO = ZoneInfo("US/Eastern")

# Set torch device
DEVICE = torch.device("cuda" if torch.cuda.device_count() >= 1 else "cpu")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    pass
