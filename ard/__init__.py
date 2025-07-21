from . import collection
from . import layout
from . import offshore
from . import wind_query
from . import utils

from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent
ASSET_DIR = BASE_DIR / "api" / "default_systems"
