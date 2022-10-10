from pathlib import Path
from epm.experiment_utils import dependencies

MAXINT = 10**10

with (Path(__file__).parent / 'requirements.txt').open() as fh:
    dependencies.verify_packages(fh.read())
