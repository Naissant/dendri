from pathlib import Path

import pytest
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructField,
    ArrayType,
    LongType,
    StringType,
    StructType,
)

from pysoma.conftest import ensure_clean_dir
from pysoma.dataframe import (
    cols_to_array,
    two_columns_to_dictionary,
    dict_to_map_literal,
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
)


def test_cols_to_array(spark_context):
    sdf = spark_context.createDataFrame(
        [("001", "11", "12", ["11", "12"]), ("002", "11", None, ["11"])],
        schema=["id", "col1", "col2", "array_col"],
    )
    res = cols_to_array(
        sdf.drop("array_col"), array_name="array_col", columns=["col1", "col2"]
    )

    assert sorted(sdf.collect()) == sorted(res.collect())


def test_two_columns_to_dictionary():
    def it_returns_a_dictionary(spark_context):
        data = [("jose", 1), ("li", 2), ("luisa", 3)]
        source_df = spark_context.createDataFrame(data, ["name", "age"])
        actual = two_columns_to_dictionary(source_df, "name", "age")
        assert {"jose": 1, "li": 2, "luisa": 3} == actual


class TestDictToMapLiteral:
    def test_dict_to_map_literal_basic(self, spark_context):
        sdf = spark_context.createDataFrame(
            [("1", "A"), ("2", "B")], schema=["id", "val"]
        )

        mapping_basic = {"A": "a", "B": "b"}
        map_literal = dict_to_map_literal(mapping_basic)

        res = sdf.withColumn("mapped", map_literal[F.col("val")])

        exp = spark_context.createDataFrame(
            [("1", "A", "a"), ("2", "B", "b")], schema=["id", "val", "mapped"]
        )

        assert sorted(res.collect()) == sorted(exp.collect())

    def test_dict_to_map_literal_nested(self, spark_context):
        sdf = spark_context.createDataFrame(
            [("1", "A"), ("2", "B")], schema=["id", "val"]
        )

        mapping_basic = {"A": ["a", "ah"], "B": ["b", "bah"]}
        map_literal = dict_to_map_literal(mapping_basic)

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
            "The ['fun'] columns are not included in the DataFrame with the following "
            "columns ['name', 'age']"
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
            "The [StructField(city,StringType,true)] StructFields are not included "
            "in the DataFrame with the following StructFields "
            "StructType(List(StructField(name,StringType,true),"
            "StructField(age,LongType,true)))"
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
            "The ['age'] columns are not allowed to be included in the DataFrame "
            "with the following columns ['name', 'age']"
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
