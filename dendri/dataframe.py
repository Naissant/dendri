import re
import json
import types
import functools
from pathlib import Path
from typing import Union, List
from collections import namedtuple

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.column import Column
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructType,
    MapType,
    BinaryType,
    BooleanType,
    DateType,
    NullType,
    NumericType,
    TimestampType,
    UserDefinedType,
)


def validate_config_path(path: Union[str, Path]):
    """
    Pending deprecation.
    """
    path = Path(path)

    # validate user-supplied path
    #   - Must end with '.config.json'
    name_parts = path.resolve().name.split(".")
    if name_parts[-1] != "json" or name_parts[-2] != "config":
        raise ValueError(
            "path to dataframe config file must end with '.config.json'\n"
            f"path: {path.resolve()}"
        )


def show_by_type(sdf: DataFrame, dtype: str, *args, **kwargs):
    """
    .show() columns of provided dtype

    Engine: Spark
    """
    columnList = [item[0] for item in sdf.dtypes if item[1].startswith(dtype)]
    sdf.select(columnList).show(*args, **kwargs)


def namedtuple_from_schema(schema: StructType, name: str):
    """
    Convert a PySpark schema into a named tuple

    Args:
        schema: PySpark schema
        name: Name of namedtuple

    Returns:
        namedtuple representation of PySpark schema
    """
    return namedtuple(name, schema.names, defaults=(None,) * len(schema.names))


def cols_to_array(*cols, remove_na: bool = True) -> Column:
    """
    Create a column of ArrayType() from user-supplied column list.

    Args:
        cols: columns to convert into array.
        remove_na (optional): Remove nulls from array. Defaults to True.

    Returns:
        Column of ArrayType()
    """
    if remove_na:
        return F.filter(F.array(*cols), lambda x: x.isNotNull())
    else:
        return F.array(*cols)


def cols_to_dict(df: DataFrame, key_col_name: str, value_col_name: str) -> dict:
    """
    Creates a dictionary from pair of columns.

    Args:
        df: PySpark DataFrame
        key_col_name: Column name to use as dict key
        value_col_name: Column name to use as dict value

    Returns:
        A dict of the two DataFrame columns
    """
    k, v = key_col_name, value_col_name
    return {x[k]: x[v] for x in df.select(k, v).collect()}


def dict_to_map(mapping: dict) -> Column:
    """
    Create a PySpark map literal from dict of k-v pairs.

    Args:
        mapping: Mapping dict

    Returns:
        PySpark Column of MapType()
    """
    literal_list = []

    for k, v in mapping.items():
        literal_list.append(F.lit(k))
        if isinstance(v, (str, int, float, bool)):
            literal_list.append(F.lit(v))
        elif isinstance(v, (list, tuple, set)):
            literal_list.append(F.array([F.lit(x) for x in v]))

    return F.create_map(literal_list)


def map_column(col_name: str, mapping: Union[Column, dict]) -> Column:
    """
    Apply mapping expression to column.

    Args:
        col_name: Name of column to map values
        mapping: MapType() column or dict with desired mapping

    Returns:
        Column with mapped values
    """
    if isinstance(mapping, dict):
        mapping = dict_to_map(mapping)

    return mapping[F.col(col_name)]


class DataFrameMissingColumnError(ValueError):
    """Raise this when there's a missing DataFrame column"""


class DataFrameMissingStructFieldError(ValueError):
    """Raise this when there's a missing DataFrame StructField"""


class DataFrameProhibitedColumnError(ValueError):
    """Raise this when a DataFrame includes prohibited columns"""


def validate_presence_of_columns(df: DataFrame, required_col_names: List[str]):
    """
    Validate DataFrame contains listed columns.

    Args:
        df: PySpark DataFrame to check columns
        required_col_names: List of column names to validate

    Raises:
        DataFrameMissingColumnError: If DataFrame is missing a required column
    """
    missing_cols = [x for x in required_col_names if x not in df.columns]

    error_message = (
        "The following columns are not included in the DataFrame:"
        f" {', '.join(missing_cols)}"
    )

    if missing_cols:
        raise DataFrameMissingColumnError(error_message)


def validate_schema(df: DataFrame, required_schema: StructType):
    """
    Raises Error if df schema does not match required schema.

    Args:
        df: PySpark DataFrame to validate schema
        required_schema: Required schema

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

    Raises:
        DataFrameMissingStructFieldError: If DataFrame is missing a StructField
    """
    missing_struct_fields = [x for x in required_schema if x not in df.schema]

    error_message = (
        "The following StructFields are not included in the DataFrame:"
        f" {missing_struct_fields}"
    )

    if missing_struct_fields:
        raise DataFrameMissingStructFieldError(error_message)


def validate_absence_of_columns(df: DataFrame, prohibited_col_names: List[str]):
    """
    Validate DataFrame does not contain listed columns.

    Args:
        df: PySpark DataFrame to check columns
        prohibited_col_names: List of column names to validate

    Raises:
        DataFrameProhibitedColumnError: If DataFrame is includes a prohibited column
    """
    extra_cols = [x for x in prohibited_col_names if x in df.columns]

    error_message = (
        "The following columns are not allowed to be included in the DataFrame:"
        f" {', '.join(extra_cols)}"
    )

    if extra_cols:
        raise DataFrameProhibitedColumnError(error_message)


def string_to_column_name(x: str) -> str:
    """
    Converts non-alphanumeric characters to "_" and lowercases string. Repeated
    non-alphanumeric characters are replaced with a single "_".

    Args:
        x: String to convert to column name

    Returns:
        Re-formatted string for using as a PySpark DataFrame column name
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
            *args, cached_file_path=cached_file_path, overwrite=overwrite, **kwargs
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


def data_template_from_schema(schema: StructType):
    """Returns a hand-fillable template based on supplied schema."""
    header_string = ""
    output = "("
    for field in schema.fields:
        if isinstance(field.dataType, StringType):
            output += '"String", '
        elif isinstance(field.dataType, DateType):
            output += 'date.fromisoformat(""), '
        elif isinstance(field.dataType, TimestampType):
            output += 'datetime.fromisoformat(""), '
        elif isinstance(field.dataType, ArrayType):
            output += '["String?"], '
        elif isinstance(field.dataType, (MapType, StructType)):
            output += '{"": ""}, '
        elif isinstance(field.dataType, BinaryType):
            output += 'b"", '
        elif isinstance(field.dataType, BooleanType):
            output += "True False, "
        elif isinstance(field.dataType, NumericType):
            output += f"{field.dataType}, "
        elif isinstance(field.dataType, NullType):
            output += "None, "
        elif isinstance(field.dataType, UserDefinedType):
            output += "UDT(), "
        else:
            # Something numeric
            output += "Error, "
        header_string += field.name + ", "
    header_string = header_string[:-2]
    output = output[:-2] + ")"
    return header_string + "\n" + output


def set_column_order(
    df: DataFrame, column_order: list, remove_unlisted: bool = False
) -> DataFrame:
    """
    Set the column order for a DataFrame to a desired order. DataFrame colums not in
    column_order will be placed at the end of the DataFrame.

    Args:
        df: DataFrame to reorder columns.
        column_order: List of column names in desired order.
        remove_unlisted: Remove columns from DataFrame if not in supplied column list.
            Default: False.

    Returns:
        DataFrame with columns in desired order.
    """

    # Type check the inputs
    if not isinstance(df, DataFrame):
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


def rename_by_dict(df: DataFrame, renames: dict) -> DataFrame:
    """
    Renames DataFrame columns according to dictionary

    Args:
        df: PySpark DataFrame
        renames: Dictionary of renames in {existing: new} format

    Returns:
        DataFrame with column names updated
    """

    col_list = []

    for column in df.columns:
        if column in renames.keys():
            col_list.append(F.col(column).alias(renames[column]))
        else:
            col_list.append(F.col(column))

    return df.select(col_list)


def outer_union_corr(*dfs: DataFrame) -> DataFrame:
    """
    Returns a new DataFrame containing the union of all rows and columns from passed
    DataFrames. This is equivalent to OUTER UNION CORR in SQL. To do a SQL-style set
    union (that does deduplication of elements), use this function followed by
    distinct(). The difference between this function and unionByName() is that this
    function returns all columns from all DataFrames.

    Args:
        dfs: Sequence of DataFrames to union

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


def col_to_set(df: DataFrame, col: str) -> set:
    """
    Returns a set from specified column of DataFrame

    Args:
        df: PySpark DataFrame
        col: Name of column

    Returns:
        A set
    """
    return set(df.select(col).distinct().toPandas()[col])


def col_to_list(df: DataFrame, col: str) -> list:
    """
    Returns a list from specified column of DataFrame

    Args:
        df: PySpark DataFrame
        col: Name of column

    Returns:
        A list
    """
    return list(df.select(col).toPandas()[col])


def array_isin(
    array_col: str, values: Union[str, list], require_all: bool = False
) -> Column:
    """
    Check if one or more array values are in user-supplied list.

    Args:
        array_col: Array column name
        values: List of values to check
        require_all (optional): If True, requires the array to contain all values.
            Defaults to False.

    Returns:
        Column of BooleanType()
    """
    if isinstance(values, str):
        condition = F.array_contains(F.col(array_col), F.lit(values))
    else:
        condition = F.array_contains(F.col(array_col), F.lit(values[0]))

    if len(values) > 1:
        for v in values[1:]:
            if require_all:
                condition = condition & (F.array_contains(F.col(array_col), F.lit(v)))
            else:
                condition = condition | (F.array_contains(F.col(array_col), F.lit(v)))

    return condition


def saveParquetTable(
    self,
    table_name: str,
    file_path: Union[str, Path],
    partition_cols: Union[List[str], str] = None,
    bucket_cols: Union[List[str], str] = None,
    bucket_size: int = None,
    sort_cols: Union[List[str], str] = None,
    write_mode="overwrite",
) -> None:
    """
    Saves a DataFrame using the saveAsTable method and writes necessary metadata to the
    parquet file as _dendri_metadata. This metadata is used when calling
    `readParquetTable` to return a DataFrame that retains information about bucketing.

    Args:
        table_name: Name of table to create
        file_path: Path to save table
        partition_cols (optional): Columns to partition by. Defaults to None.
        bucket_cols (optional): Columns to bucket by. Defaults to None.
        bucket_size (optional): Number of buckets per partition. Defaults to None.
        sort_cols (optional): Columns to sort each bucket by. Only supports ascending
            order. Defaults to None.
        write_mode (optional): Mode to use when calling `saveAsTable`. Defaults to
            "overwrite".
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)

    if partition_cols is not None:
        save_builder = self.repartition(partition_cols).write.partitionBy(
            partition_cols
        )
    else:
        save_builder = self.write

    if bucket_cols is not None:
        save_builder = save_builder.bucketBy(bucket_size, bucket_cols)

    if sort_cols is not None:
        save_builder = save_builder.sortBy(sort_cols)

    save_builder.option("path", file_path).saveAsTable(
        table_name, format="parquet", mode=write_mode
    )

    # Create dictionary of metadata
    meta_dict = {
        "partition_cols": partition_cols,
        "bucket_cols": bucket_cols,
        "bucket_size": bucket_size,
        "sort_cols": sort_cols,
        "schema": self.dtypes,
    }

    # Write metadata to parquet file
    with open(f"{file_path}/_dendri_metadata", "w") as f:
        json.dump(meta_dict, f, indent=4)


def _convert_dtypes_to_str(x: list) -> str:
    return ", ".join([f"{i[0]} {i[1]}" for i in x])


def _convert_list_to_str(x: Union[str, list]) -> str:
    if isinstance(x, list):
        x = ", ".join(x)
    else:
        x = str(x)

    return x


def _sql_builder(
    file: str,
    table: str,
    schema: str,
    partition_cols: str,
    bucket_cols: str,
    bucket_size: str,
    sort_cols: str,
) -> str:
    create_table = f"CREATE TABLE IF NOT EXISTS {table} ({schema}) USING PARQUET "

    if partition_cols != "None":
        partition_by = f"PARTITIONED BY ({partition_cols}) "
    else:
        partition_by = ""

    if bucket_cols != "None":
        clustered_by = f"CLUSTERED BY ({bucket_cols}) "
    else:
        clustered_by = ""

    if sort_cols != "None":
        sorted_by = f"SORTED BY ({sort_cols}) "
    else:
        sorted_by = ""

    if bucket_size != "None":
        into_buckets = f"INTO {bucket_size} BUCKETS "
    else:
        into_buckets = ""

    location = f"LOCATION '{file}'"

    sql = (
        create_table + partition_by + clustered_by + sorted_by + into_buckets + location
    )

    return sql


def readParquetTable(self, file_path: Union[str, Path], table_name: str) -> DataFrame:
    """
    Reads parquet files using the PySpark Table API to return a DataFrame with bucketing
    information retained. Bucketing info is only kept when writing the DataFrame using
    the `saveParquetTable` method.

    Args:
        file_path: Path to parquet file
        table_name: Name of table to create when reading parquet file

    Returns:
        DataFrame
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)

    meta_file = f"{file_path}/_dendri_metadata"

    if Path(meta_file).exists():
        with open(meta_file, "r") as f:
            meta_dict = json.load(f)

        # Extract metadata and convert to strings needed for CREATE TABLE SQL
        partition_cols = _convert_list_to_str(meta_dict["partition_cols"])
        bucket_cols = _convert_list_to_str(meta_dict["bucket_cols"])
        bucket_size = _convert_list_to_str(meta_dict["bucket_size"])
        sort_cols = _convert_list_to_str(meta_dict["sort_cols"])
        schema = _convert_dtypes_to_str(meta_dict["schema"])

        # Build the CREATE TABLE SQL
        sql = _sql_builder(
            file_path,
            table_name,
            schema,
            partition_cols,
            bucket_cols,
            bucket_size,
            sort_cols,
        )

        self.sql(sql)

        # Need to register partitions with Hive manually
        if partition_cols != "None":
            self.sql(f"MSCK REPAIR TABLE {table_name}")

        df = self.table(table_name)
    else:
        df = self.read.parquet(file_path)

    return df


DataFrame.saveParquetTable = saveParquetTable
SparkSession.readParquetTable = readParquetTable


def get_dataframe(obj):
    """Attempts to return a Spark DataFrame from supplied reference."""
    spark = SparkSession.getActiveSession()
    if isinstance(obj, str):
        return spark.read.parquet(obj)
    if isinstance(obj, Path):
        return spark.read.parquet(str(obj))
    if isinstance(obj, DataFrame):
        return obj
    if isinstance(obj, types.FunctionType):
        return obj()
    if isinstance(obj, types.MethodType):
        # NOTE(Wes): Not 100% sure on this one.
        return obj()
    # Return whatever it is and let the end user figure it out (None, perhaps?)
    return obj
