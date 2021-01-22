from collections import namedtuple

import pyspark.sql.functions as F
from pyspark.sql.types import StructType


def namedtuple_from_schema(schema: StructType, name: str):
    """Convert a PySpark schema into a named tuple."""
    return namedtuple(name, schema.names, defaults=(None,) * len(schema.names))
