"""Rebuild current figures. Earlier artwork is preserved in figures/v1."""
from pathlib import Path
import runpy
here = Path(__file__).resolve().parent
for script in ['redesign_figures.py', 'make_architecture.py']:
    runpy.run_path(str(here / script), run_name='__main__')
