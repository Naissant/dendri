from typing import Union, List
from datetime import date

from dateutil.relativedelta import relativedelta
from pyspark.sql import DataFrame, Column
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, BooleanType, DateType, ShortType, FloatType

from dendri.dataframe import set_column_order


def condense_segments(
    df: DataFrame,
    group_col: Union[str, List[str]],
    start_dt_col: str,
    end_dt_col: str,
    tolerance: int = 1,
) -> DataFrame:
    """
    Combines overlapping start and end dates according to the allowable tolerance for
    each group.

    Args:
        df: Name of dataframe to condense.
        group_col: Columns to group by from df.
        start_dt_col: Name of the start date column from df.
        end_dt_col: Name of the end date column from df.
        tolerance: The allowable number of days between segments.

    Returns:
        Input DataFrame with updated user-supplied start_dt_col, end_dt_col
    """

    # Treat group_col as list in all cases
    if isinstance(group_col, str):
        group_col = [group_col]

    # windowSpec defining group_col + start/end dates
    # - used to determine which segments are overlapping
    win_spec_overlap = (
        Window.partitionBy([F.col(x) for x in group_col])
        .orderBy(F.col(start_dt_col), F.col(end_dt_col))
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    # windowSpec defining group_col + overlapping segments (defined by "_group_id")
    # - used to determine min/max dates across "_group_id"
    win_spec_condense = Window.partitionBy(
        [F.col(x) for x in (group_col + ["_group_id"])]
    )

    return (
        df.withColumn(
            "_group_id",
            F.when(
                F.col(start_dt_col)
                <= F.max(F.date_add(F.col(end_dt_col), tolerance)).over(
                    win_spec_overlap
                ),
                None,
            ).otherwise(F.monotonically_increasing_id()),
        )
        .withColumn(
            "_group_id",
            F.last(F.col("_group_id"), ignorenulls=True).over(
                win_spec_overlap.rowsBetween(Window.unboundedPreceding, 0)
            ),
        )
        .withColumn("_new_start", F.min(F.col(start_dt_col)).over(win_spec_condense))
        .withColumn("_new_end", F.max(F.col(end_dt_col)).over(win_spec_condense))
        .drop(start_dt_col, end_dt_col, "_group_id")
        .withColumnRenamed("_new_start", start_dt_col)
        .withColumnRenamed("_new_end", end_dt_col)
        .orderBy([x for x in (group_col + [start_dt_col] + [end_dt_col])])
    )


def extend_segments(
    df: DataFrame,
    group_col: Union[str, List[str]],
    start_dt_col: str,
    end_dt_col: str,
    tolerance: int = 1,
    retain_shape: bool = True,
) -> DataFrame:
    """
    Extends overlapping start and end dates according to the allowable tolerance for
    each group.

    Args:
        df: Name of dataframe with extendable segments.
        group_col: Columns to group by from df.
        start_dt_col: Name of the start date column from df.
        end_dt_col: Name of the end date column from df.
        tolerance: The allowable number of days between segments.
        retain_shape: Retain original column and row counts.
            Default: True
            False will return distinct `group_col`, `start_dt_col`, `end_dt_col`.

    Returns:
        Input DataFrame with updated user-supplied start_dt_col, end_dt_col
    """

    def extend_array_segments(date_segment_array, tolerance: int):
        output_dates = []
        for idx, dates in enumerate(date_segment_array):
            # First segment: init temp date variables and segment days
            if idx == 0:
                rolling_start_dt = dates[0]
                rolling_end_dt = dates[1]
                rolling_days = (rolling_end_dt - rolling_start_dt).days + 1
            # 2nd+ segment: check if overlaps with rollingious segment
            elif dates[0] <= rolling_end_dt + relativedelta(days=tolerance):
                # If overlaps: update rolling segment by adding current segment's days
                current_segment_days = (dates[1] - dates[0]).days + 1
                rolling_days = rolling_days + current_segment_days
                rolling_end_dt = rolling_start_dt + relativedelta(days=rolling_days - 1)
            else:  # This is the start of a segment which does not overlap
                # insert current rolling dates into output_dates
                output_dates.extend([[rolling_start_dt, rolling_end_dt]])

                # Re-initialize dates & days
                rolling_start_dt = dates[0]
                rolling_end_dt = dates[1]
                rolling_days = (rolling_end_dt - rolling_start_dt).days + 1

        # insert last segment into output_dates
        output_dates.extend([[rolling_start_dt, rolling_end_dt]])

        return output_dates

    extend_array_segments_udf = F.udf(
        extend_array_segments, ArrayType(ArrayType(DateType()))
    )

    # Treat group_col as list in all cases
    if isinstance(group_col, str):
        group_col = [group_col]

    extended_segments_sdf = (
        df.groupBy(group_col)
        .agg(
            F.array_sort(F.collect_list(F.array(start_dt_col, end_dt_col))).alias(
                "_input_segments_array"
            ),
        )
        .withColumn(
            "_new_segments",
            extend_array_segments_udf("_input_segments_array", F.lit(tolerance)),
        )
        .select(*group_col, F.explode("_new_segments").alias("_new_segments"))
        .select(
            *group_col,
            F.col("_new_segments")[0].alias(start_dt_col),
            F.col("_new_segments")[1].alias(end_dt_col),
        )
    )

    if retain_shape:
        retained_sdf = (
            df.join(
                extended_segments_sdf.select(
                    *group_col,
                    F.col(start_dt_col).alias("_start_dt_new"),
                    F.col(end_dt_col).alias("_end_dt_new"),
                ),
                group_col,
                "left",
            )
            .filter(
                F.col(start_dt_col).between(
                    F.col("_start_dt_new"), F.col("_end_dt_new")
                )
            )
            .drop(start_dt_col, end_dt_col)
            .withColumnRenamed("_start_dt_new", start_dt_col)
            .withColumnRenamed("_end_dt_new", end_dt_col)
        )
        return set_column_order(retained_sdf, df.columns)
    else:
        return extended_segments_sdf


def covered_days(
    df: DataFrame,
    group_col: Union[str, List[str]],
    segment_start_dt_col: Union[str, Column],
    segment_end_dt_col: Union[str, Column],
    window_start_dt: Union[date, Column],
    window_end_dt: Union[date, Column],
    covered_days_col_name: str = "covered_days",
) -> DataFrame:
    """
    Count days covered by segments within provided winow. Days covered is calculated
    without condensing or expanding overlapping segments.

    Args:
        df: Name of dataframe with extendable segments.
        group_col: Columns to group by from df.
        segment_start_dt_col: Segment start date column name.
        segment_end_dt_col: Segment end date column name.
        window_start_dt: Start date of window in which to count covered days.
        window_end_dt: End date of window in which to count covered days.
        covered_days_col_name: Name of column with number of days covered.

    Returns:
        DataFrame:
        {
            ... original columns ...
            `covered_days_col_name` (LongType)
        }
    """

    # group_col always treated like list
    if isinstance(group_col, str):
        group_col = [group_col]

    # convert to F.col() if str was supplied
    if isinstance(segment_start_dt_col, str):
        segment_start_dt_col = F.col(segment_start_dt_col)
    if isinstance(segment_end_dt_col, str):
        segment_end_dt_col = F.col(segment_end_dt_col)

    # convert supplied date to F.lit() if supplied
    if isinstance(window_start_dt, date):
        window_start_dt = F.lit(window_start_dt)
    if isinstance(window_end_dt, date):
        window_end_dt = F.lit(window_end_dt)

    # windowSpec defining group_col + start/end dates
    # - used to determine which segments are overlapping
    win_spec_overlap = (
        Window.partitionBy([F.col(x) for x in group_col])
        .orderBy(segment_start_dt_col, segment_end_dt_col)
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    # windowSpec defining group_col + overlapping segments (defined by "_group_id")
    # - used to determine min/max dates across "_group_id"
    win_spec_condense = Window.partitionBy(
        [F.col(x) for x in (group_col + ["_group_id"])]
    ).orderBy(group_col)

    # windowSpec defining group_col + overlapping segments (defined by "_group_id")
    # - used to determine min/max dates across "_group_id"
    winSpec_first_record = Window.partitionBy(
        [F.col(x) for x in (group_col + ["_group_id_first"] + ["_group_id"])]
    )

    winSpec_group_col = Window.partitionBy(group_col)

    sdf = (
        df.withColumn(
            "_group_id",
            F.when(
                segment_start_dt_col
                <= F.max(F.date_add(segment_end_dt_col, 1)).over(win_spec_overlap),
                None,
            ).otherwise(F.monotonically_increasing_id()),
        )
        .withColumn(
            "_group_id",
            F.last(F.col("_group_id"), ignorenulls=True).over(
                win_spec_overlap.rowsBetween(Window.unboundedPreceding, 0)
            ),
        )
        .withColumn(
            "_group_id_start_dt", F.min(segment_start_dt_col).over(win_spec_condense)
        )
        .withColumn(
            "_group_id_end_dt", F.max(segment_end_dt_col).over(win_spec_condense)
        )
        # Find days covered in specified date range
        .withColumn(
            "_group_id_covered_days",
            F.datediff(
                F.least(F.col("_group_id_end_dt"), window_end_dt),
                F.greatest(F.col("_group_id_start_dt"), window_start_dt),
            )
            + 1,
        )
        # Segments outside window will produce -negative values, zero them
        .withColumn(
            "_group_id_covered_days",
            F.when(F.col("_group_id_covered_days") < 0, 0).otherwise(
                F.col("_group_id_covered_days")
            ),
        )
        # Flag only first record of _group_id
        .withColumn(
            "_group_id_first",
            F.when(F.row_number().over(win_spec_condense) == 1, True).otherwise(False),
        )
        # Sum across only the first segment of each _group_id
        .withColumn(
            covered_days_col_name,
            F.sum(F.col("_group_id_covered_days")).over(winSpec_first_record),
        )
        .withColumn(
            covered_days_col_name,
            F.sum(
                F.when(
                    F.col("_group_id_first"), F.col(covered_days_col_name)
                ).otherwise(0)
            ).over(winSpec_group_col),
        )
    ).drop(
        "_group_id",
        "_group_id_start_dt",
        "_group_id_end_dt",
        "_group_id_covered_days",
        "_group_id_first",
    )

    return sdf


def first_event_in_x_days(
    df: DataFrame,
    group_col: Union[str, List[str]],
    start_dt_col: str,
    days: int,
) -> DataFrame:
    """
    Identify the first valid event in an x-day period. Identify invalid events that are
    within x-days of the last valid event.

    Args:
        df: Input DataFrame
        group_col: Columns to group by from input DataFrame
        start_dt_col: Start date column from input DataFrame
        days: Number of days to check between events

    Returns:
        Input DataFrame with BooleanType column 'valid_event' for showing if the event
        was the first in an x-day period.
            Columns: {
                group_col
                start_dt_col (DateType())
                valid_event  (BooleanType())
            }
    """

    def first_event_func(dates, tolerance):
        comp_dt = dates[0]
        keep_dt = []
        for i, dt in enumerate(dates):
            if i == 0:
                keep_dt.append(True)
            elif dt <= comp_dt + relativedelta(days=tolerance):
                keep_dt.append(False)
            else:
                keep_dt.append(True)
                comp_dt = dt

        return keep_dt

    first_event_udf = F.udf(first_event_func, ArrayType(BooleanType()))

    if isinstance(group_col, str):
        group_col = [group_col]

    return (
        df.groupBy(group_col)
        # Convert long column to sorted array per group
        .agg(F.array_sort(F.collect_list(start_dt_col)).alias("_temp_dt"))
        # Iterate over array per group instead of iterating over RDD rows
        .withColumn("_temp_status", first_event_udf("_temp_dt", F.lit(days)))
        .select(
            *group_col,
            F.explode(F.arrays_zip("_temp_dt", "_temp_status")).alias("_temp_map"),
        )
        .select(
            *group_col,
            F.col("_temp_map")["_temp_dt"].alias(start_dt_col),
            F.col("_temp_map")["_temp_status"].alias("valid_event"),
        )
    )


def count_events_by_period(
    df: DataFrame,
    group_col: Union[str, List[str]],
    count_col: str,
    dt_col: str,
    max_dt: Union[str, date],
    period: str,
    num_periods: int,
    count_unique: bool = False,
) -> DataFrame:
    """
    Count the number of events per period per group.

    Args:
        df: Name of input DataFrame
        group_col: Name of column(s) to group by when counting
        count_col: Name of column to count
        dt_col: Name of date column for determining periods
        max_dt: End date of final period
        period: Period type to use. Accepted values are 'years', 'months', 'days', and
            'weeks'
        num_periods: Number of period to count (going backwards from max_dt)
        count_unique: Count unique values. Defaults to False.

    Returns:
        Number of events per period per group.
            Columns: {
                group_col
                event_cnt_prd0
                ...
                event_cnt_prd{num_periods - 1}
            }
    """

    if isinstance(max_dt, str):
        max_dt = date.fromisoformat(max_dt)

    end_dt = max_dt

    period_col_list = []
    period = period.lower()

    # Create columns for each period populated with the count col
    for i in range(1, num_periods + 1):
        if period in ("y", "year", "years"):
            offset = relativedelta(years=i)
        if period in ("m", "months", "month"):
            offset = relativedelta(months=i)
        elif period in ("d", "day", "days"):
            offset = relativedelta(days=i)
        elif period in ("w", "weeks", "week"):
            offset = relativedelta(weeks=i)
        else:
            raise ValueError(
                "period only supports the values of 'years', 'months', 'days', and "
                "'weeks'"
            )

        start_dt = max_dt - offset + relativedelta(days=1)
        col_name = f"event_cnt_prd{i - 1}"
        period_col_list.append(col_name)

        df = df.withColumn(
            col_name,
            F.when(
                F.col(dt_col).between(F.lit(start_dt), F.lit(end_dt)), F.col(count_col)
            ),
        )

        end_dt = max_dt - offset

    if count_unique:
        agg_func = [
            F.countDistinct(col).cast(ShortType()).alias(col) for col in period_col_list
        ]
    else:
        agg_func = [
            F.count(col).cast(ShortType()).alias(col) for col in period_col_list
        ]

    return df.groupBy(group_col).agg(*agg_func)


def age(
    dob_col: str,
    age_dt: Union[str, date, Column],
    floor_sw: bool = True,
) -> Column:
    """
    Calculate age.

    Args:
        dob_col: Column name containing date of birth.
        age_dt: Date to calculate age.
        floor_sw: Return age as whole number, rounded down. Defaults to True.

    Returns:
        Age as of the specified date. Returns FloatType if round is False. Otherwise
            returns ShortType.
    """
    if isinstance(age_dt, str):
        age_dt = date.fromisoformat(age_dt)

    if isinstance(age_dt, date):
        age_dt = F.lit(age_dt)

    age = (F.months_between(age_dt, dob_col) / 12).cast(FloatType())

    if floor_sw:
        age = F.floor(age).cast(ShortType())

    return age
