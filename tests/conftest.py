import tempfile
from shutil import rmtree
from contextlib import contextmanager

import pytest

from dendri.utils import spark_getOrCreate, quiet_py4j


@pytest.fixture(scope="session")
def spark_context(request):
    """
    fixture for creating a spark sql session

    Args:
        request: pytest.FixtureRequest object
    """
    spark_context = spark_getOrCreate(
        app_name="pytest-spark-fixture",
        config={
            "spark.sql.shuffle.partitions": 1,
            "spark.default.parallelism": 1,
            "spark.rdd.compress": False,
            "spark.shuffle.compress": False,
            # Note(Wes): I didn't see any decrease in runtime after making the below
            #   changes, though they're listed as helpful on the pytest-spark repo:
            #   https://github.com/malexer/pytest-spark
            #   Keeping the options commented out and available as future options
            # "spark.dynamicAllocation.enabled": "false",
            # "spark.executor.cores": 1,
            # "spark.executor.instances": 1,
            # "spark.io.compression.codec": "lz4",
            # "spark.sql.catalogImplementation": "hive",
        },
    )
    quiet_py4j()
    try:
        yield spark_context
    finally:
        spark_context.stop()


@contextmanager
def ensure_clean_dir():
    """
    Get a temporary directory path and agrees to remove on close.
    Yields
    ------
    Temporary directory path
    """
    directory_name = tempfile.mkdtemp(suffix="")
    try:
        yield directory_name
    finally:
        try:
            rmtree(directory_name)
        except OSError:
            pass
