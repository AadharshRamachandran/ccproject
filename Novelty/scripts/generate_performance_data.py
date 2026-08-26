"""Novelty-local entry point for the labelled synthetic development data."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[2]/'Paper'/'scripts'/'generate_performance_data.py'),run_name='__main__')
