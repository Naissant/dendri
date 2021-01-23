from datetime import date
from dataclasses import dataclass
from typing import List

import pytest
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ByteType,
    DateType,
    LongType,
)

from pysoma.dataframe import namedtuple_from_schema


def d(s):
    return date.fromisoformat(s)


@dataclass
class EntityDateSegmentData:
    """Class organizes date segments for a specific Entity."""

    date_segment_data: list

    def date_segments(self):
        return self.date_segment_data


@dataclass
class AllDateSegmentData:
    """Class organizes date segments across all Entities."""

    _entities: List[EntityDateSegmentData]

    def date_segments(self):
        date_segments = []
        for entity in self._entities:
            date_segments.extend(entity.date_segments())
        return date_segments


date_segment_schema = StructType(
    [
        StructField("entity_id", StringType(), False),
        StructField("iterator", ByteType(), True),
        StructField("start_dt", DateType(), True),
        StructField("end_dt", DateType(), True),
    ]
)

date_segment_tuple = namedtuple_from_schema(date_segment_schema, "date_segment_tuple")

entity_id = "001"
entity_1 = EntityDateSegmentData(
    date_segment_data=[
        date_segment_tuple(
            entity_id=entity_id,
            iterator=1,
            start_dt=d("2020-01-01"),
            end_dt=d("2020-01-03"),
        ),
        date_segment_tuple(
            entity_id=entity_id,
            iterator=2,
            start_dt=d("2020-01-02"),
            end_dt=d("2020-01-04"),
        ),
    ]
)
