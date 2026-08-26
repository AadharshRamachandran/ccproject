"""Novelty-local entry point for the shared World Cup preparation procedure."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[2]/'Paper'/'scripts'/'prepare_worldcup98.py'),run_name='__main__')
