# Make modules available
from . import dataframe
from . import dates
from . import utils

# Shortcut specific functionality
from dendri.dataframe import SparkSession, DataFrame

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dendri")
except PackageNotFoundError:
    __version__ = "unknown"
