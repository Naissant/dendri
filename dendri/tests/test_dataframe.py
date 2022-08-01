from pathlib import Path
import json

import pytest
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructField,
    ArrayType,
    LongType,
    StringType,
    StructType,
    MapType,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    FractionalType,
    IntegerType,
    IntegralType,
    NullType,
    NumericType,
    TimestampType,
    UserDefinedType,
    ShortType,
)

from dendri.conftest import ensure_clean_dir
from dendri.dataframe import (
    cols_to_array,
    cols_to_dict,
    dict_to_map,
    DataFrameMissingColumnError,
    DataFrameMissingStructFieldError,
    DataFrameProhibitedColumnError,
    validate_presence_of_columns,
    validate_schema,
    validate_absence_of_columns,
    string_to_column_name,
    stage_dataframe_to_disk,
    set_column_order,
    rename_by_dict,
    outer_union_corr,
    array_isin,
    validate_config_path,
    # show_by_type,
    get_dataframe,
    data_template_from_schema,
    melt,
    merge_map_arrays,
)


def test_validate_config_path():
    validate_config_path(Path("df.config.json"))


def test_validate_config_path_raises():
    with pytest.raises(ValueError):
        validate_config_path("df.json.config")
    with pytest.raises(ValueError):
        validate_config_path("df.json.config")
    with pytest.raises(ValueError):
        validate_config_path("json.config")
    with pytest.raises(ValueError):
        validate_config_path("df._.json")
    with pytest.raises(ValueError):
        validate_config_path("df.config._")


@pytest.mark.skip(reason="Not tested...")
def test_show_by_type():
    pass


class TestColsToArray:

    input_data = [
        ("001", "11", "12", ["11", "12"]),
        ("002", "11", None, ["11"]),
    ]
    input_schema = ["id", "col1", "col2", "array_col"]

    def test_cols_to_array_cols_as_args(self, spark_context):
        sdf = spark_context.createDataFrame(
            data=self.input_data, schema=self.input_schema
        )
        res = sdf.withColumn("array_col", cols_to_array("col1", "col2"))

        assert sorted(sdf.collect()) == sorted(res.collect())

    def test_cols_to_array_cols_as_list(self, spark_context):
        sdf = spark_context.createDataFrame(
            data=self.input_data, schema=self.input_schema
        )
        res = sdf.withColumn("array_col", cols_to_array(["col1", "col2"]))

        assert sorted(sdf.collect()) == sorted(res.collect())

    def test_cols_to_array_cols_as_set(self, spark_context):
        sdf = spark_context.createDataFrame(
            data=self.input_data, schema=self.input_schema
        )
        res = sdf.withColumn("array_col", F.array_sort(cols_to_array({"col1", "col2"})))

        assert sorted(sdf.collect()) == sorted(res.collect())

    def test_cols_to_array_single_col(self, spark_context):
        sdf = spark_context.createDataFrame(
            data=[("001", "11", "12", ["12"]), ("002", "11", None, [])],
            schema=self.input_schema,
        )
        res = sdf.withColumn("array_col", cols_to_array("col2"))

        assert sorted(sdf.collect()) == sorted(res.collect())

    def test_cols_to_array_does_not_remove_duplicates(self, spark_context):
        sdf = spark_context.createDataFrame(
            data=[
                ("1", 1, 2, 3, 4, [1, 2, 3, 4]),
                ("2", 1, 1, 1, 1, [1, 1, 1, 1]),
                ("3", 1, 2, None, None, [1, 2]),
                ("4", 1, 1, None, None, [1, 1]),
            ],
            schema=StructType(
                [
                    StructField("id", StringType()),
                    StructField("col1", LongType()),
                    StructField("col2", LongType()),
                    StructField("col3", LongType()),
                    StructField("col4", LongType()),
                    StructField("col_arr", ArrayType(LongType())),
                ]
            ),
        )

        res = sdf.withColumn(
            "col_arr", cols_to_array("col1", "col2", "col3", "col4", remove_na=True)
        )

        assert sorted(sdf.collect()) == sorted(res.collect())

    def test_cols_to_array_accepts_Column_type(self, spark_context):
        sdf = spark_context.createDataFrame(
            data=self.input_data, schema=self.input_schema
        )
        res = sdf.withColumn("array_col", cols_to_array(F.col("col1"), F.col("col2")))

        assert sorted(sdf.collect()) == sorted(res.collect())

    def test_cols_to_array_accepts_column_with_maps(self, spark_context):
        sdf = spark_context.createDataFrame(
            [
                ("1", 1, 2, {"a": 5}, [1, 5]),
                ("2", 1, 2, {"a": 6}, [1, 6]),
                ("3", 1, 2, {"b": 7}, [1])
                # black ..............
            ],
            StructType(
                [
                    StructField("id", StringType()),
                    StructField("col1", LongType()),
                    StructField("col2", LongType()),
                    StructField("col3", MapType(StringType(), LongType())),
                    StructField("array_col", ArrayType(LongType())),
                ]
            ),
        )

        res = sdf.withColumn("array_col", cols_to_array("col1", F.col("col3")["a"]))

        assert sorted(sdf.collect()) == sorted(res.collect())


def test_cols_to_dict():
    def it_returns_a_dictionary(spark_context):
        data = [("jose", 1), ("li", 2), ("luisa", 3)]
        source_df = spark_context.createDataFrame(data, ["name", "age"])
        actual = cols_to_dict(source_df, "name", "age")
        assert {"jose": 1, "li": 2, "luisa": 3} == actual


class TestDictToMap:
    def test_dict_to_map_basic(self, spark_context):
        sdf = spark_context.createDataFrame(
            [("1", "A"), ("2", "B")], schema=["id", "val"]
        )

        mapping_basic = {"A": "a", "B": "b"}
        map_literal = dict_to_map(mapping_basic)

        res = sdf.withColumn("mapped", map_literal[F.col("val")])

        exp = spark_context.createDataFrame(
            [("1", "A", "a"), ("2", "B", "b")], schema=["id", "val", "mapped"]
        )

        assert sorted(res.collect()) == sorted(exp.collect())

    def test_dict_to_map_nested(self, spark_context):
        sdf = spark_context.createDataFrame(
            [("1", "A"), ("2", "B")], schema=["id", "val"]
        )

        mapping_basic = {"A": ["a", "ah"], "B": ["b", "bah"]}
        map_literal = dict_to_map(mapping_basic)

        res = sdf.withColumn("mapped", map_literal[F.col("val")])

        exp = spark_context.createDataFrame(
            [("1", "A", ["a", "ah"]), ("2", "B", ["b", "bah"])],
            schema=StructType(
                [
                    StructField("id", StringType(), False),
                    StructField("val", StringType(), False),
                    StructField("mapped", ArrayType(StringType(), False), False),
                ]
            ),
        )

        assert sorted(res.collect()) == sorted(exp.collect())


class TestValidatePresenceOfColumns:
    def test_it_raises_if_a_required_column_is_missing(self, spark_context):
        data = [("jose", 1), ("li", 2), ("luisa", 3)]
        source_df = spark_context.createDataFrame(data, ["name", "age"])
        with pytest.raises(DataFrameMissingColumnError) as excinfo:
            validate_presence_of_columns(source_df, ["name", "age", "fun"])
        assert excinfo.value.args[0] == (
            "The following columns are not included in the DataFrame: fun"
        )

    def test_it_does_nothing_if_all_required_columns_are_present(self, spark_context):
        data = [("jose", 1), ("li", 2), ("luisa", 3)]
        source_df = spark_context.createDataFrame(data, ["name", "age"])
        validate_presence_of_columns(source_df, ["name"])


class TestValidateSchema:
    def test_it_raises_when_struct_field_is_missing1(self, spark_context):
        data = [("jose", 1), ("li", 2), ("luisa", 3)]
        source_df = spark_context.createDataFrame(data, ["name", "age"])
        required_schema = StructType(
            [
                StructField("name", StringType(), True),
                StructField("city", StringType(), True),
            ]
        )
        with pytest.raises(DataFrameMissingStructFieldError) as excinfo:
            validate_schema(source_df, required_schema)
        assert excinfo.value.args[0] == (
            "The following StructFields are not included in the DataFrame:"
            " [StructField('city', StringType(), True)]"
        )

    def test_it_does_nothing_when_the_schema_matches(self, spark_context):
        data = [("jose", 1), ("li", 2), ("luisa", 3)]
        source_df = spark_context.createDataFrame(data, ["name", "age"])
        required_schema = StructType(
            [
                StructField("name", StringType(), True),
                StructField("age", LongType(), True),
            ]
        )
        validate_schema(source_df, required_schema)


class TestValidateAbsenseOfColumns:
    def test_it_raises_when_a_unallowed_column_is_present(self, spark_context):
        data = [("jose", 1), ("li", 2), ("luisa", 3)]
        source_df = spark_context.createDataFrame(data, ["name", "age"])
        with pytest.raises(DataFrameProhibitedColumnError) as excinfo:
            validate_absence_of_columns(source_df, ["age", "cool"])
        assert excinfo.value.args[0] == (
            "The following columns are not allowed to be included in the DataFrame: age"
        )

    def test_it_does_nothing_when_no_unallowed_columns_are_present(self, spark_context):
        data = [("jose", 1), ("li", 2), ("luisa", 3)]
        source_df = spark_context.createDataFrame(data, ["name", "age"])
        validate_absence_of_columns(source_df, ["favorite_color"])


def test_string_to_column_name():
    start_end_tuple = [
        (
            "Adverse Effect of Beta-Adrenoreceptor Antagonists",
            "adverse_effect_of_beta_adrenoreceptor_antagonists",
        ),
        (
            "Rotavirus Vaccine (2 Dose Schedule) Administered",
            "rotavirus_vaccine_2_dose_schedule_administered",
        ),
        (
            "Partial Hospitalization/Intensive Outpatient",
            "partial_hospitalization_intensive_outpatient",
        ),
        ("HbA1c Level 7.0-9.0", "hba1c_level_7_0_9_0"),
    ]

    expected = [x[1] for x in start_end_tuple]
    result = [string_to_column_name(x[0]) for x in start_end_tuple]

    assert sorted(expected) == sorted(result)


def test_stage_dataframe_to_disk_function(spark_context):

    with ensure_clean_dir() as tmpdir:

        tmpdir = Path(tmpdir)
        tmp_parquet_path = tmpdir / "tmp.parquet"

        @stage_dataframe_to_disk(tmp_parquet_path)
        def tmp_sdf():
            data = [(1,), (2,)]
            sdf = spark_context.createDataFrame(data, ["col"])
            return sdf

        tmp_sdf()
        tmp_sdf(overwrite=True)

        assert tmp_parquet_path.exists()
        assert isinstance(
            spark_context.read.parquet(str(tmp_parquet_path)), F.DataFrame
        )


def test_stage_dataframe_to_disk_method(spark_context):

    with ensure_clean_dir() as tmpdir:

        tmpdir = Path(tmpdir)
        tmp_parquet_path = tmpdir / "tmp.parquet"

        class TestProcess:
            @stage_dataframe_to_disk(tmp_parquet_path)
            def tmp_sdf(self):
                data = [(1,), (2,)]
                sdf = spark_context.createDataFrame(data, ["col"])
                return sdf

        TestProcess().tmp_sdf()
        TestProcess().tmp_sdf(overwrite=True)

        assert tmp_parquet_path.exists()
        assert isinstance(
            spark_context.read.parquet(str(tmp_parquet_path)), F.DataFrame
        )


def test_set_column_order(spark_context):
    data = [
        (1, 2, 3),
    ]
    schema = ["c1", "c2", "c3"]
    sdf = spark_context.createDataFrame(data, schema)

    # full set of columns
    _ = set_column_order(sdf, ["c3", "c2", "c1"])
    assert _.columns == ["c3", "c2", "c1"]

    # subset of columns
    _ = set_column_order(sdf, ["c2", "c1"])
    assert _.columns == ["c2", "c1", "c3"]

    # remove_unlisted = True
    _ = set_column_order(sdf, ["c3", "c2"], remove_unlisted=True)
    assert _.columns == ["c3", "c2"]


def test_rename_by_dict(spark_context):
    sdf = spark_context.createDataFrame([(1, 2, 3)], ["col1", "col2", "col3"])

    sdf = rename_by_dict(sdf, {"col1": "_col1", "col4": "_col4"})

    assert sdf.columns == ["_col1", "col2", "col3"]


def test_outer_union_corr(spark_context):
    sdf1 = spark_context.createDataFrame([(1, 2, 3)], ["col1", "col2", "col3"])
    sdf2 = spark_context.createDataFrame([(4, 5, 6)], ["col1", "col4", "col5"])

    expected_output = spark_context.createDataFrame(
        [(1, 2, 3, None, None), (4, None, None, 5, 6)],
        ["col1", "col2", "col3", "col4", "col5"],
    )

    res = outer_union_corr(sdf1, sdf2).select("col1", "col2", "col3", "col4", "col5")

    assert sorted(res.collect()) == sorted(expected_output.collect())


class TestArrayIsin:
    def test_array_isin(self, spark_context):
        sdf = spark_context.createDataFrame(
            [(1, 2, 3, True), (4, 5, 6, False)], ["col1", "col2", "col3", "col4"]
        ).select(F.array("col1", "col2", "col3").alias("array_col"), "col4")

        res = sdf.drop("col4").select(array_isin("array_col", [1, 7, 8]))
        exp = sdf.drop("array_col")

        assert sorted(res.collect()) == sorted(exp.collect())

    def test_array_isin_requires_all(self, spark_context):
        sdf = spark_context.createDataFrame(
            [(1, 2, 3, True), (4, 5, 6, False)],
            ["col1", "col2", "col3", "col4"],
        ).select(F.array("col1", "col2", "col3").alias("array_col"), "col4")

        res = sdf.drop("col4").select(array_isin("array_col", [1, 3], True))
        exp = sdf.drop("array_col")

        assert sorted(res.collect()) == sorted(exp.collect())


def test_saveParquetTable(spark_context):
    with ensure_clean_dir() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_parquet_path = tmpdir / "tmp.parquet"

        # Create test df
        df = spark_context.createDataFrame(
            data=[(1, 2, 3), (1, 2, 4), (2, 3, 1), (3, 3, 1)],
            schema=["col1", "col2", "col3"],
        )

        # Write df
        df.saveParquetTable(
            table_name="tmp",
            file_path=str(tmp_parquet_path),
            partition_cols="col1",
            bucket_cols="col2",
            bucket_size=1,
            sort_cols="col3",
        )

        # Expected metatdata
        exp_meta = {
            "partition_cols": "col1",
            "bucket_cols": "col2",
            "bucket_size": 1,
            "sort_cols": "col3",
            "schema": [["col1", "bigint"], ["col2", "bigint"], ["col3", "bigint"]],
        }

        assert tmp_parquet_path.exists()
        assert isinstance(
            spark_context.read.parquet(str(tmp_parquet_path)), F.DataFrame
        )

        # Acutal metadata
        with open(tmp_parquet_path / "_dendri_metadata", "r") as f:
            res_meta = json.load(f)

        assert res_meta == exp_meta

        # Delete PySpark table
        spark_context.sql("DROP TABLE tmp")


def test_readParquetTable(spark_context):
    with ensure_clean_dir() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_parquet_path = tmpdir / "tmp.parquet"

        # Create test df
        exp_df = spark_context.createDataFrame(
            data=[(1, 2, 3), (1, 2, 4), (2, 3, 1), (3, 3, 1)],
            schema=["col1", "col2", "col3"],
        )

        # Write df
        exp_df.saveParquetTable(
            table_name="tmp",
            file_path=str(tmp_parquet_path),
            partition_cols="col1",
            bucket_cols="col2",
            bucket_size=1,
            sort_cols="col3",
        )

        # Delete PySpark table
        spark_context.sql("DROP TABLE tmp")

        # Read df
        res_df = spark_context.readParquetTable(
            table_name="tmp", file_path=str(tmp_parquet_path)
        ).select("col1", "col2", "col3")

        assert sorted(res_df.collect()) == sorted(exp_df.collect())

        # Delete PySpark table
        spark_context.sql("DROP TABLE tmp")


def test_readParquetTable_no_bucket(spark_context):
    with ensure_clean_dir() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_parquet_path = tmpdir / "tmp.parquet"

        # Create test df
        exp_df = spark_context.createDataFrame(
            data=[(1, 2, 3), (1, 2, 4), (2, 3, 1), (3, 3, 1)],
            schema=["col1", "col2", "col3"],
        )

        # Write df
        exp_df.saveParquetTable(
            table_name="tmp",
            file_path=str(tmp_parquet_path),
            partition_cols="col1",
        )

        # Delete PySpark table
        spark_context.sql("DROP TABLE tmp")

        # Read df
        res_df = spark_context.readParquetTable(
            table_name="tmp", file_path=str(tmp_parquet_path)
        ).select("col1", "col2", "col3")

        assert sorted(res_df.collect()) == sorted(exp_df.collect())

        # Delete PySpark table
        spark_context.sql("DROP TABLE tmp")


class TestGetDataFrame:

    dataframe = {
        "data": [(1, 2, 3), (1, 2, 4), (2, 3, 1), (3, 3, 1)],
        "schema": ["col1", "col2", "col3"],
    }

    def test_get_dataframe_str(self, spark_context):
        sdf = spark_context.createDataFrame(**self.dataframe)
        with ensure_clean_dir() as dir:
            dir = Path(dir)
            sdf_path = dir / "sdf.parquet"
            sdf.write.parquet(str(sdf_path))
            res = get_dataframe(str(sdf_path))

            assert sorted(res.collect()) == sorted(sdf.collect())

    def test_get_dataframe_Path(self, spark_context):
        sdf = spark_context.createDataFrame(**self.dataframe)
        with ensure_clean_dir() as dir:
            dir = Path(dir)
            sdf_path = dir / "sdf.parquet"
            sdf.write.parquet(str(sdf_path))
            res = get_dataframe(sdf_path)

            assert sorted(res.collect()) == sorted(sdf.collect())

    def test_get_dataframe_Path_saveParquetTable(self, spark_context):
        sdf = spark_context.createDataFrame(**self.dataframe)
        with ensure_clean_dir() as dir:
            dir = Path(dir)

            # Save DataFrame using saveParquetTable, with buckets and partitions
            sdf_path = dir / "sdf.parquet"
            sdf.saveParquetTable(
                table_name="tmp",
                file_path=str(sdf_path),
                partition_cols="col1",
                bucket_cols="col2",
                bucket_size=1,
                sort_cols="col3",
            )

            # Read DataFrame back into Spark
            # Additional select necessary for comparison
            # - partitioned columns are moved to the far right of the DataFrame on read
            res = get_dataframe(sdf_path).select(self.dataframe["schema"])

            assert sorted(res.collect()) == sorted(sdf.collect())

    def test_get_dataframe_DataFrame(self, spark_context):
        sdf = spark_context.createDataFrame(**self.dataframe)
        res = get_dataframe(sdf)
        assert isinstance(res, DataFrame)

    def test_get_dataframe_FunctionType(self, spark_context):
        sdf = spark_context.createDataFrame(**self.dataframe)

        def function_that_returns_DataFrame() -> DataFrame:
            return spark_context.createDataFrame(**self.dataframe)

        res = get_dataframe(function_that_returns_DataFrame)
        assert sorted(sdf.collect()) == sorted(res.collect())

    def test_get_dataframe_MethodType(self, spark_context):
        sdf = spark_context.createDataFrame(**self.dataframe)

        class ClassContainingMethods:
            dataframe = self.dataframe

            def method_returns_DataFrame(self):
                return spark_context.createDataFrame(**self.dataframe)

        instance = ClassContainingMethods()
        res = get_dataframe(instance.method_returns_DataFrame)
        assert sorted(res.collect()) == sorted(sdf.collect())

    def test_get_dataframe_MethodType_staticmethod(self, spark_context):
        sdf = spark_context.createDataFrame(**self.dataframe)

        class ClassContainingMethods:
            dataframe = self.dataframe

            @staticmethod
            def method_returns_DataFrame():
                return spark_context.createDataFrame(**self.dataframe)

        res = get_dataframe(ClassContainingMethods.method_returns_DataFrame)
        assert sorted(res.collect()) == sorted(sdf.collect())

        instance = ClassContainingMethods()
        res = get_dataframe(instance.method_returns_DataFrame)
        assert sorted(res.collect()) == sorted(sdf.collect())

    def test_get_dataframe_MethodType_classmethod(self, spark_context):
        sdf = spark_context.createDataFrame(**self.dataframe)

        class ClassContainingMethods:
            dataframe = self.dataframe

            @classmethod
            def method_returns_DataFrame(cls):
                return spark_context.createDataFrame(**cls.dataframe)

        res = get_dataframe(ClassContainingMethods.method_returns_DataFrame)
        assert sorted(res.collect()) == sorted(sdf.collect())

        instance = ClassContainingMethods()
        res = get_dataframe(instance.method_returns_DataFrame)
        assert sorted(res.collect()) == sorted(sdf.collect())


def test_data_template_from_schema(spark_context):
    schema = StructType(
        [
            StructField("ArrayType(String)", ArrayType(StringType(), True), True),
            StructField("BinaryType", BinaryType(), True),
            StructField("BooleanType", BooleanType(), True),
            StructField("ByteType", ByteType(), True),
            StructField("DateType", DateType(), True),
            StructField("DecimalType", DecimalType(), True),
            StructField("DoubleType", DoubleType(), True),
            StructField("FloatType", FloatType(), True),
            StructField("FractionalType", FractionalType(), True),
            StructField("IntegerType", IntegerType(), True),
            StructField("IntegralType", IntegralType(), True),
            StructField("LongType", LongType(), True),
            StructField("MapType", MapType(StringType(), StringType()), True),
            StructField("NullType", NullType(), True),
            StructField("NumericType", NumericType(), True),
            StructField("StringType", StringType(), True),
            StructField("StructType", StructType(), True),
            StructField("TimestampType", TimestampType(), True),
            StructField("UserDefinedType", UserDefinedType(), True),
            StructField("ShortType", ShortType(), True),
        ]
    )

    exp = (
        "ArrayType(String), BinaryType, BooleanType, ByteType, DateType, DecimalType, "
        "DoubleType, FloatType, FractionalType, IntegerType, IntegralType, LongType, "
        "MapType, NullType, NumericType, StringType, StructType, TimestampType, "
        'UserDefinedType, ShortType\n(["String?"], b"", True False, ByteType(), '
        'date.fromisoformat(""), DecimalType(10,0), DoubleType(), FloatType(), '
        'FractionalType(), IntegerType(), IntegralType(), LongType(), {"": ""}, None, '
        'NumericType(), "String", {"": ""}, datetime.fromisoformat(""), UDT(), '
        "ShortType())"
    )
    res = data_template_from_schema(schema)
    assert res == exp


class TestMelt:
    input_data = [
        ("001", 1, 2),
        ("002", 2, 3),
        ("002", 3, 4),
    ]
    input_schema = ["id", "col1", "col2"]

    def test_melt_with_id(self, spark_context):
        sdf = spark_context.createDataFrame(
            data=self.input_data, schema=self.input_schema
        )
        res = melt(sdf, melt_cols=["col1", "col2"], id_cols="id")
        exp = spark_context.createDataFrame(
            [
                ("001", "col1", 1),
                ("001", "col2", 2),
                ("002", "col1", 2),
                ("002", "col2", 3),
                ("002", "col1", 3),
                ("002", "col2", 4),
            ],
            ["id", "col", "val"],
        )

        assert sorted(exp.collect()) == sorted(res.collect())

    def test_melt_no_id(self, spark_context):
        sdf = spark_context.createDataFrame(
            data=self.input_data, schema=self.input_schema
        )
        res = melt(sdf, melt_cols=["col1", "col2"])
        exp = spark_context.createDataFrame(
            [
                ("col1", 1),
                ("col2", 2),
                ("col1", 2),
                ("col2", 3),
                ("col1", 3),
                ("col2", 4),
            ],
            ["col", "val"],
        )

        assert sorted(exp.collect()) == sorted(res.collect())

    def test_melt_one_col(self, spark_context):
        sdf = spark_context.createDataFrame(
            data=self.input_data, schema=self.input_schema
        )
        res = melt(sdf, melt_cols="col2")
        exp = spark_context.createDataFrame(
            [("col2", 2), ("col2", 3), ("col2", 4)], ["col", "val"]
        )

        assert sorted(exp.collect()) == sorted(res.collect())


def test_merge_map_arrays(spark_context):
    sdf = spark_context.createDataFrame(
        data=[
            ("01", {"a": [0, 1]}, {"a": [2, 3]}),
            ("02", {"b": [0, 1]}, {"c": [0, 1]}),
            ("03", {"d": [0, 1]}, {"d": [0, 1]}),
            ("04", {"e": [0, 1]}, {"e": []}),
            ("02", {"f": [0, 1], "g": [0, 1]}, {"f": [2, 3], "g": [2, 3]}),
        ],
        schema=["id", "map1", "map2"],
    )
    res = sdf.withColumn("merge", merge_map_arrays("map1", "map2"))
    exp = spark_context.createDataFrame(
        [
            ("01", {"a": [0, 1]}, {"a": [2, 3]}, {"a": [0, 1, 2, 3]}),
            ("02", {"b": [0, 1]}, {"c": [0, 1]}, {"b": [0, 1], "c": [0, 1]}),
            ("03", {"d": [0, 1]}, {"d": [0, 1]}, {"d": [0, 1]}),
            ("04", {"e": [0, 1]}, {"e": []}, {"e": [0, 1]}),
            (
                "02",
                {"f": [0, 1], "g": [0, 1]},
                {"f": [2, 3], "g": [2, 3]},
                {"f": [0, 1, 2, 3], "g": [0, 1, 2, 3]},
            ),
        ],
        ["id", "map1", "map2", "merge"],
    )

    # NOTE(Riley): The sorted function was not working because the MapTypeColumn returns
    # a dictionary via .collect() and '<' is not supported between instances of dicts
    assert exp.collect() == res.collect()
