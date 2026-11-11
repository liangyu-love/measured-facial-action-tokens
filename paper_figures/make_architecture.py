"""Current architecture entry point; earlier design is in figures/v2."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('make_mechanism_figure.py')), run_name='__main__')
