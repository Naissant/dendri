# Make modules available
from . import dataframe
from . import dates
from . import utils

# Shortcut specific functionality
from dendri.dataframe import SparkSession, DataFrame

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
