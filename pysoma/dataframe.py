from typing import Union, List
from pathlib import Path
import functools
import re
from collections import namedtuple

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType


def namedtuple_from_schema(schema: StructType, name: str):
    """Convert a PySpark schema into a named tuple."""
    return namedtuple(name, schema.names, defaults=(None,) * len(schema.names))


def cols_to_array(
    df, array_name: str, columns: Union[str, List[str]], remove_na: bool = True
):
    """
    Returns an array named `array_name` containing the values from provided `columns`.

    Args:
        df (pyspark.sql.DataFrame)
    """
    array_column = F.array([F.col(x) for x in columns])

    if remove_na:
        return df.withColumn(
            array_name, F.array_except(array_column, F.array(F.lit(None)))
        )
    else:
        return df.withColumn(array_name, array_column)


def two_columns_to_dictionary(df, key_col_name, value_col_name):
    """
    Creates dict from pair of columns.
    """
    k, v = key_col_name, value_col_name
    return {x[k]: x[v] for x in df.select(k, v).collect()}


def dict_to_map_literal(mapping: dict):
    """
    Create Spark map literal from dict of k-v pairs.
    """
    # create_map requires a list of values
    # [1, 2, 3, 4] will map 1=>2 and 3=>4
    # ["A", "a", "B", "b"] will map "A"->"a" and "B"->"b"
    # ["A", ["ah", "bah"]] will map "A"->array("ah", "bah")
    # All literal values need to be wrapped with F.lit(literal-value)
    literal_list = []
    for k, v in mapping.items():
        literal_list.append(F.lit(k))
        if isinstance(v, (str, int, float)):
            literal_list.append(F.lit(v))
        elif isinstance(v, list):
            literal_list.append(F.array([F.lit(x) for x in v]))
    return F.create_map(literal_list)


def map_column(column, mapping_expression):
    """
    Apply mapping expression to column.
    """
    return mapping_expression[F.col(column)]


class DataFrameMissingColumnError(ValueError):
    """raise this when there's a DataFrame column error"""


class DataFrameMissingStructFieldError(ValueError):
    """raise this when there's a DataFrame column error"""


class DataFrameProhibitedColumnError(ValueError):
    """raise this when a DataFrame includes prohibited columns"""


def validate_presence_of_columns(df, required_col_names):
    """
    Validate DataFrame contains listed columns.
    """
    all_col_names = df.columns
    missing_col_names = [x for x in required_col_names if x not in all_col_names]
    error_message = (
        f"The {missing_col_names} columns are not included in the DataFrame "
        f"with the following columns {all_col_names}"
    )
    if missing_col_names:
        raise DataFrameMissingColumnError(error_message)


def validate_schema(df, required_schema):
    """
    Raises Error if df schema does not match required schema.

    Args:
        df: pyspark.sql.DataFrame
        required_schema: [pyspark.sql.types.StructField, ...]

    Example:
        data = [("jose", 1), ("li", 2), ("luisa", 3)]
        source_df = spark.createDataFrame(data, ["name", "age"])
        required_schema = StructType(
            [
                StructField("name", StringType(), True),
                StructField("city", StringType(), True),
            ]
        )
        # Raises Error, as "city" does not exist, but "age" does.
        validate_schema(source_df, required_schema)
    """
    all_struct_fields = df.schema
    missing_struct_fields = [x for x in required_schema if x not in all_struct_fields]
    error_message = (
        f"The {missing_struct_fields} StructFields are not included in the "
        f"DataFrame with the following StructFields {all_struct_fields}"
    )
    if missing_struct_fields:
        raise DataFrameMissingStructFieldError(error_message)


def validate_absence_of_columns(df, prohibited_col_names: list):
    """
    Validate DataFrame does not contain listed columns.
    """
    all_col_names = df.columns
    extra_col_names = [x for x in all_col_names if x in prohibited_col_names]
    error_message = (
        f"The {extra_col_names} columns are not allowed to be included in the "
        f"DataFrame with the following columns {all_col_names}"
    )
    if extra_col_names:
        raise DataFrameProhibitedColumnError(error_message)


def string_to_column_name(x: str):
    """
    Converts non-alphanumeric characters to "_" and lowercases string.

    Repeated non-alphanumeric characters are replaced with a single "_".
    """
    return re.sub("[^0-9a-zA-Z]+", "_", x).lower()


def stage_dataframe_to_disk(
    cached_file_path: Union[str, Path], overwrite: bool = False
):
    """
    Stage returned DataFrame to disk. Decorated function must return a Spark DataFrame.

    Args:
        cached_file_path: path to desired parquet location on disk.
        overwrite: overwrite existing parquet if exists.

    Example
    -------
    ```
    @stage_dataframe_to_disk("path-to-file/file.parquet")
    def very_important_sdf():
        data = [(1,), (2,)]
        sdf = spark.createDataFrame(data, ["col"])
        return sdf

    # DataFrame is written to disk @ path-to-file/file.parquet
    very_important_sdf()
    ```
    """

    def decorator_to_disk(func):
        @functools.wraps(func)
        def wrapper_to_disk(
            *args, cached_file_path=cached_file_path, overwrite=overwrite, **kwargs,
        ):
            cached_file_path = Path(cached_file_path)
            spark = SparkSession.getActiveSession()
            cached_file_path = Path(cached_file_path)
            if cached_file_path.exists() is False or overwrite is True:
                func(*args, **kwargs).write.parquet(
                    str(cached_file_path), mode="overwrite"
                )
            return spark.read.parquet(str(cached_file_path))

        return wrapper_to_disk

    return decorator_to_disk


def set_column_order(
    df: F.DataFrame, column_order: list, remove_unlisted: bool = False,
):
    """
    Set the column order for a DataFrame to a desired order.
    DataFrame colums not in column_order will be placed at the end of the DataFrame.

    Args:
        df: DataFrame to reorder columns.
        column_order: List of column names in desired order.

    Optional Args:
        remove_unlisted: Remove columns from DataFrame if not in supplied column list.
            Default: False.

    Returns:
        DataFrame: DataFrame with columns in desired order.
    """

    # Type check the inputs
    if not isinstance(df, (F.DataFrame)):
        raise TypeError("df must be a Spark DataFrame.")
    if not isinstance(column_order, list):
        raise TypeError("column_order must be a list" f"{column_order} was supplied")
    if not isinstance(remove_unlisted, bool):
        raise TypeError("remove_unlisted must be a bool")

    # List of columns in both df & column_order
    ordered_columns_intersect = [col for col in column_order if col in set(df.columns)]

    # columns in df that were not provided to function
    in_df_not_provided = [col for col in df.columns if col not in set(column_order)]

    if remove_unlisted is False:
        ordered_columns = ordered_columns_intersect + in_df_not_provided
    elif remove_unlisted is True:
        ordered_columns = ordered_columns_intersect

    return df.select(ordered_columns)


def rename_by_dict(sdf, renames: dict):
    """Renames DataFrame columns if they exist as keys in `renames`"""

    # TODO(WES): Using .select([loop-through-select-or-rename]) will save some CPU

    cols_to_rename = [x for x in sdf.columns if x in renames.keys()]
    for col in cols_to_rename:
        sdf = sdf.withColumnRenamed(col, renames[col])
    return sdf


def outer_union_corr(*dfs: DataFrame) -> DataFrame:
    """
    Returns a new DataFrame containing the union of all rows and columns from passed
    DataFrames. This is equivalent to OUTER UNION CORR in SQL. To do a SQL-style set
    union (that does deduplication of elements), use this function followed by
    distinct(). The difference between this function and unionByName() is that this
    function returns all columns from all DataFrames.

    Args:
        dfs (DataFrame): Sequence of DataFrames to union

    Returns:
        DataFrame
    """
    # Identify all unique column names from all dfs
    all_cols = set.union(*[set(i.columns) for i in dfs])

    for i, df in enumerate(dfs):
        # Identify missing columns from each df
        missing_cols = all_cols.difference(set(df.columns))

        # Assign missing columns to each df with null values
        __df = df.select("*", *[F.lit(None).alias(col) for col in missing_cols])

        # For first df, reassign to _df. Otherwise overwrite and union with _df
        if i == 0:
            _df = __df
        else:
            _df = _df.unionByName(__df)

    return _df
