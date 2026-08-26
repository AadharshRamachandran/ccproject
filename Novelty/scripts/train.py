"""Structurally parallel training entry point for novelty artifacts."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('train_novelty.py')),run_name='__main__')
