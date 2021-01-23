import logging

from pyspark.sql import SparkSession


def spark_getOrCreate(
    driver_memory: str = "1g",
    app_name: str = "pyspark Session",
    timezone: str = "UTC",
    aqe: bool = True,
    port: str = "4040",
    config: dict = None,
) -> SparkSession:
    """
    Gets an existing SparkSession or, if there is no existing one, creates a new one
    based on the options set in this function.

    Args:
        driver_memory (str, optional): Amount of memory to use for the driver process,
            i.e. where SparkContext is initialized, in the same format as JVM memory
            strings with a size unit suffix ("k", "m", "g" or "t") (e.g. 512m, 2g).
            Defaults to "1g".
        app_name (str, optional): The name of your application. This will appear in the
            UI and in log data. Defaults to "pyspark Session".
        timezone (str, optional): The ID of session local timezone in the format of
            either region-based zone IDs or zone offsets. This format is also passed to
            the "spark.driver.extraJavaOptions" and "spark.executor.extraJavaOptions".
            Defaults to "UTC".
        aqe (bool, optional): When true, enable adaptive query execution, which
            re-optimizes the query plan in the middle of query execution, based on
            accurate runtime statistics. Defaults to True.
        port (str, optional): Port for your application's dashboard, which shows memory
            and workload data. Defaults to "4040".
        config (dict, optional): Sets a config option. Options set here will overwrite
            any defaults. Defaults to None.

    Returns:
        SparkSession
    """

    sess = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", driver_memory)
        # -Duser.timezone required in some situations - saving as notes
        .config("spark.driver.extraJavaOptions", f"-Duser.timezone={timezone}")
        .config("spark.executor.extraJavaOptions", f"-Duser.timezone={timezone}")
        .config("spark.sql.session.timeZone", f"{timezone}")
        .config("spark.sql.adaptive.enabled", aqe)
        .config("spark.ui.port", port)
    )

    # User-supplied configs will overwrite default values
    if config is not None:
        for k, v in config.items():
            sess = sess.config(k, v)

    return sess.getOrCreate()


def quiet_py4j():
    """turn down spark logging"""
    logger = logging.getLogger("py4j")
    logger.setLevel(logging.WARN)
