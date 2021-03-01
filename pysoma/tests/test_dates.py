from datetime import date

import pytest
import pyspark.sql.functions as F

from pysoma.dates import (
    condense_segments,
    extend_segments,
    covered_days,
    first_event_in_x_days,
    count_events_by_period,
    age,
)


def d(s):
    return date.fromisoformat(s)


class TestCondenseSegments:
    input_df = (
        [
            # Scenario #1: Non-overlap segment and segments 0 days apart
            ("001", "100", d("2017-01-01"), d("2017-01-02")),
            ("001", "101", d("2018-01-01"), d("2018-01-15")),
            ("001", "102", d("2018-01-15"), d("2018-01-20")),
            # Scenario #2: Segments 1 days apart
            ("002", "200", d("2018-02-01"), d("2018-02-15")),
            ("002", "201", d("2018-02-16"), d("2018-02-20")),
            # Scenario #3: Segments 2 days apart
            ("003", "300", d("2018-03-01"), d("2018-03-15")),
            ("003", "301", d("2018-03-17"), d("2018-03-20")),
            # Scenario #4: Segments 1 day apart with nested segment
            ("004", "400", d("2019-08-01"), d("2019-08-07")),
            ("004", "401", d("2019-08-05"), d("2019-08-06")),
            ("004", "402", d("2019-08-08"), d("2019-08-15")),
        ],
        ["entity_id", "iterator", "start_dt", "end_dt"],
    )

    output_df_tolerance_1 = (
        [
            ("001", "100", d("2017-01-01"), d("2017-01-02")),
            ("001", "101", d("2018-01-01"), d("2018-01-20")),
            ("001", "102", d("2018-01-01"), d("2018-01-20")),
            ("002", "200", d("2018-02-01"), d("2018-02-20")),
            ("002", "201", d("2018-02-01"), d("2018-02-20")),
            ("003", "300", d("2018-03-01"), d("2018-03-15")),
            ("003", "301", d("2018-03-17"), d("2018-03-20")),
            ("004", "400", d("2019-08-01"), d("2019-08-15")),
            ("004", "401", d("2019-08-01"), d("2019-08-15")),
            ("004", "402", d("2019-08-01"), d("2019-08-15")),
        ],
        ["entity_id", "iterator", "start_dt", "end_dt"],
    )

    output_df_tolerance_2 = (
        [
            ("001", "100", d("2017-01-01"), d("2017-01-02")),
            ("001", "101", d("2018-01-01"), d("2018-01-20")),
            ("001", "102", d("2018-01-01"), d("2018-01-20")),
            ("002", "200", d("2018-02-01"), d("2018-02-20")),
            ("002", "201", d("2018-02-01"), d("2018-02-20")),
            ("003", "300", d("2018-03-01"), d("2018-03-20")),
            ("003", "301", d("2018-03-01"), d("2018-03-20")),
            ("004", "400", d("2019-08-01"), d("2019-08-15")),
            ("004", "401", d("2019-08-01"), d("2019-08-15")),
            ("004", "402", d("2019-08-01"), d("2019-08-15")),
        ],
        ["entity_id", "iterator", "start_dt", "end_dt"],
    )

    @pytest.mark.parametrize(
        "df, group_col, start_dt_col, end_dt_col, tolerance, expected_output",
        [
            (
                input_df,
                "entity_id",
                "start_dt",
                "end_dt",
                1,
                output_df_tolerance_1,
            ),
            (
                input_df,
                "entity_id",
                "start_dt",
                "end_dt",
                2,
                output_df_tolerance_2,
            ),
            (
                input_df,
                ["entity_id"],
                "start_dt",
                "end_dt",
                1,
                output_df_tolerance_1,
            ),
            (
                input_df,
                ["entity_id", "iterator"],
                "start_dt",
                "end_dt",
                1,
                input_df,
            ),
        ],
    )
    def test_condense_segments(
        self,
        spark_context,
        df,
        group_col,
        start_dt_col,
        end_dt_col,
        tolerance,
        expected_output,
    ):
        res = condense_segments(
            spark_context.createDataFrame(*df),
            group_col,
            start_dt_col,
            end_dt_col,
            tolerance,
        )

        exp = spark_context.createDataFrame(*expected_output)

        assert sorted(res.collect()) == sorted(exp.collect())


class TestExtendSegments:
    input_df = (
        [
            # Scenario #1: Non-overlapping sement and segments 0 days apart
            ("001", "100", d("2017-01-01"), d("2017-01-02")),
            ("001", "101", d("2018-01-01"), d("2018-01-15")),
            ("001", "102", d("2018-01-15"), d("2018-01-20")),
            # Scenario #2: Segments 1 day apart
            ("002", "200", d("2018-02-01"), d("2018-02-15")),
            ("002", "201", d("2018-02-16"), d("2018-02-20")),
            # Scenario #3: Segments 2 days apart
            ("003", "300", d("2018-03-01"), d("2018-03-15")),
            ("003", "301", d("2018-03-17"), d("2018-03-20")),
            # Scenario #4: Segments 1 day apart with nested segment
            ("004", "400", d("2019-08-01"), d("2019-08-07")),
            ("004", "401", d("2019-08-05"), d("2019-08-06")),
            ("004", "402", d("2019-08-08"), d("2019-08-15")),
            # Scenario #5: Duplicate segments and segments 2 days apart
            ("005", "500", d("2019-01-01"), d("2019-01-02")),
            ("005", "501", d("2019-01-01"), d("2019-01-02")),
            ("005", "502", d("2019-01-04"), d("2019-01-05")),
        ],
        ["entity_id", "iterator", "start_dt", "end_dt"],
    )

    output_df = (
        [
            ("001", "100", d("2017-01-01"), d("2017-01-02")),
            ("001", "101", d("2018-01-01"), d("2018-01-21")),
            ("001", "102", d("2018-01-01"), d("2018-01-21")),
            ("002", "200", d("2018-02-01"), d("2018-02-20")),
            ("002", "201", d("2018-02-01"), d("2018-02-20")),
            ("003", "300", d("2018-03-01"), d("2018-03-15")),
            ("003", "301", d("2018-03-17"), d("2018-03-20")),
            ("004", "400", d("2019-08-01"), d("2019-08-17")),
            ("004", "401", d("2019-08-01"), d("2019-08-17")),
            ("004", "402", d("2019-08-01"), d("2019-08-17")),
            ("005", "500", d("2019-01-01"), d("2019-01-06")),
            ("005", "501", d("2019-01-01"), d("2019-01-06")),
            ("005", "502", d("2019-01-01"), d("2019-01-06")),
        ],
        ["entity_id", "iterator", "start_dt", "end_dt"],
    )

    @pytest.mark.parametrize(
        "df, group_col, start_dt_col, end_dt_col, tolerance, expected_output",
        [
            (
                input_df,
                "entity_id",
                "start_dt",
                "end_dt",
                1,
                output_df,
            ),
            (
                input_df,
                ["entity_id"],
                "start_dt",
                "end_dt",
                1,
                output_df,
            ),
            (
                input_df,
                ["entity_id", "iterator"],
                "start_dt",
                "end_dt",
                1,
                input_df,
            ),
        ],
    )
    def test_extend_segments(
        self,
        spark_context,
        df,
        group_col,
        start_dt_col,
        end_dt_col,
        tolerance,
        expected_output,
    ):
        res = extend_segments(
            spark_context.createDataFrame(*df),
            group_col,
            start_dt_col,
            end_dt_col,
            tolerance,
        )

        exp = spark_context.createDataFrame(*expected_output)

        assert sorted(res.collect()) == sorted(exp.collect())


class TestCoveredDays:
    input_df = (
        [
            # Straight forward case
            (
                "1",
                d("2019-01-01"),
                d("2019-01-01"),
                d("2019-01-01"),
                d("2019-01-02"),
                1,
            ),
            # Overlapping segments
            (
                "2",
                d("2019-01-01"),
                d("2019-01-02"),
                d("2019-01-01"),
                d("2019-01-03"),
                3,
            ),
            (
                "2",
                d("2019-01-02"),
                d("2019-01-03"),
                d("2019-01-01"),
                d("2019-01-03"),
                3,
            ),
            # 1-day gap
            (
                "3",
                d("2019-01-01"),
                d("2019-01-02"),
                d("2019-01-01"),
                d("2019-01-04"),
                4,
            ),
            (
                "3",
                d("2019-01-03"),
                d("2019-01-04"),
                d("2019-01-01"),
                d("2019-01-04"),
                4,
            ),
            # 2-day gap
            (
                "4",
                d("2019-01-01"),
                d("2019-01-02"),
                d("2019-01-01"),
                d("2019-01-05"),
                4,
            ),
            (
                "4",
                d("2019-01-04"),
                d("2019-01-05"),
                d("2019-01-01"),
                d("2019-01-05"),
                4,
            ),
            # Nested Segments
            (
                "5",
                d("2019-01-01"),
                d("2019-01-04"),
                d("2019-01-01"),
                d("2019-01-05"),
                5,
            ),
            (
                "5",
                d("2019-01-02"),
                d("2019-01-03"),
                d("2019-01-01"),
                d("2019-01-05"),
                5,
            ),
            (
                "5",
                d("2019-01-04"),
                d("2019-01-05"),
                d("2019-01-01"),
                d("2019-01-05"),
                5,
            ),
            # Segment entirely before window
            (
                "6",
                d("2019-01-01"),
                d("2019-01-01"),
                d("2019-01-02"),
                d("2019-01-02"),
                0,
            ),
            # Segment entirely after window
            (
                "7",
                d("2019-01-04"),
                d("2019-01-05"),
                d("2019-01-01"),
                d("2019-01-03"),
                0,
            ),
            # Segment wider than window
            (
                "8",
                d("2019-01-01"),
                d("2019-01-05"),
                d("2019-01-02"),
                d("2019-01-03"),
                2,
            ),
            # Two segments overlapping edge of window
            (
                "9",
                d("2019-01-01"),
                d("2019-01-02"),
                d("2019-01-02"),
                d("2019-01-04"),
                2,
            ),
            (
                "9",
                d("2019-01-04"),
                d("2019-01-05"),
                d("2019-01-02"),
                d("2019-01-04"),
                2,
            ),
        ],
        [
            "entity_id",
            "segment_start_dt",
            "segment_end_dt",
            "window_start_dt",
            "window_end_dt",
            "covered_days",
        ],
    )

    def test_covered_days_window_columns(self, spark_context):

        input_df = spark_context.createDataFrame(*self.input_df)

        res = covered_days(
            df=input_df.drop("covered_days"),
            group_col="entity_id",
            segment_start_dt_col="segment_start_dt",
            segment_end_dt_col="segment_end_dt",
            window_start_dt=F.col("window_start_dt"),
            window_end_dt=F.col("window_end_dt"),
        )

        assert sorted(res.collect()) == sorted(input_df.collect())

    def test_covered_days_window_dates(self, spark_context):

        input_df = spark_context.createDataFrame(*self.input_df).filter(
            F.col("entity_id") == "5"
        )

        res = covered_days(
            df=input_df.drop("covered_days"),
            group_col="entity_id",
            segment_start_dt_col="segment_start_dt",
            segment_end_dt_col="segment_end_dt",
            window_start_dt=d("2019-01-01"),
            window_end_dt=d("2019-01-05"),
        )

        assert sorted(res.collect()) == sorted(input_df.collect())


def test_first_event_in_x_days(spark_context):
    input_df = spark_context.createDataFrame(
        [
            ("001", d("2019-01-01"), True),
            ("001", d("2019-01-15"), False),
            ("001", d("2019-01-22"), True),
            ("002", d("2019-01-25"), True),
            ("003", d("2019-03-01"), True),
            ("003", d("2019-04-01"), True),
            ("003", d("2019-05-01"), True),
            ("004", d("2019-06-01"), True),
            ("004", d("2019-06-01"), False),
            ("004", d("2019-06-22"), True),
            ("005", d("2019-01-01"), True),
            ("005", d("2019-01-22"), True),
            ("005", d("2019-01-25"), False),
        ],
        ["entity_id", "start_dt", "valid_event"],
    )

    res = first_event_in_x_days(
        df=input_df.drop("valid_event"),
        group_col="entity_id",
        start_dt_col="start_dt",
        days=20,
    )

    assert sorted(res.collect()) == sorted(input_df.collect())


def test_count_events_by_period(
    spark_context,
):
    input_df = spark_context.createDataFrame(
        [
            ("001", "A", d("2018-01-01")),
            ("002", "B", d("2018-01-01")),
            ("002", "B", d("2018-05-01")),
            ("002", "C", d("2018-01-01")),
            ("003", "D", d("2019-01-01")),
            ("004", "E", d("2018-01-01")),
            ("004", "E", d("2018-03-01")),
            ("004", "E", d("2019-05-01")),
            ("004", "F", d("2018-01-01")),
        ],
        ["entity_id", "event_id", "start_dt"],
    )

    res = count_events_by_period(
        df=input_df,
        group_col="entity_id",
        count_col="event_id",
        dt_col="start_dt",
        max_dt="2018-03-31",
        period="M",
        num_periods=3,
        count_unique=False,
    )

    exp = spark_context.createDataFrame(
        [("001", 0, 0, 1), ("002", 0, 0, 2), ("003", 0, 0, 0), ("004", 1, 0, 2)],
        ["entity_id", "event_cnt_prd0", "event_cnt_prd1", "event_cnt_prd2"],
    )

    assert sorted(res.collect()) == sorted(exp.collect())


class TestAge:
    input_df_static = (
        [
            ("001", d("2004-02-28")),
            ("002", d("2004-02-29")),
            ("003", d("2004-03-01")),
        ],
        ["entity_id", "birth_dt"],
    )

    output_df_feb28 = (
        [
            ("001", 5),
            ("002", 5),
            ("003", 4),
        ],
        ["entity_id", "age"],
    )

    output_df_mar01 = (
        [
            ("001", 5),
            ("002", 5),
            ("003", 5),
        ],
        ["entity_id", "age"],
    )

    output_df_feb28_leap = (
        [
            ("001", 8),
            ("002", 7),
            ("003", 7),
        ],
        ["entity_id", "age"],
    )

    output_df_mar01_leap = (
        [
            ("001", 8),
            ("002", 8),
            ("003", 7),
        ],
        ["entity_id", "age"],
    )

    @pytest.mark.parametrize(
        "df, age_dt, floor_sw, expected_output",
        [
            (input_df_static, "2009-02-28", True, output_df_feb28),
            (input_df_static, "2009-03-01", True, output_df_mar01),
            (input_df_static, "2012-02-28", True, output_df_feb28_leap),
            (input_df_static, "2012-02-29", True, output_df_mar01_leap),
        ],
    )
    def test_age(self, spark_context, df, age_dt, floor_sw, expected_output):
        res = spark_context.createDataFrame(*df).select(
            "entity_id", age("birth_dt", age_dt, floor_sw).alias("age")
        )
        exp = spark_context.createDataFrame(*expected_output)

        assert sorted(res.collect()) == sorted(exp.collect())
