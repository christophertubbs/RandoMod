"""
Defines classes and tools used to read and interpret USGS RDB files
"""
from __future__ import annotations

import io
import os
import pathlib
import typing
import re
from collections.abc import ValuesView
from collections.abc import ItemsView
from collections.abc import KeysView
from dataclasses import dataclass
from dataclasses import field

import pandas
import pytz
import requests

from dateutil.parser import parse as parse_date
from dateutil.tz import gettz
from pandas._typing import ReadCsvBuffer

from onlyday import Day

SITE_TIMESERIES_ROW: re.Pattern = re.compile(r"# Data provided for site (?P<site_code>\d+)\s*")
PARAMETER_CODE_PATTERN: re.Pattern = re.compile(r"#\s+\d+\s+(?P<pcode>\d{5})\s+(?P<description>.+)")
SITE_CODE_PATTERN: re.Pattern = re.compile(r"#\s+[a-zA-Z_]+\s+(?P<site_code>\d+)\s+(?P<site_name>.+)")
TS_ID_PREFIX_PATTERN = re.compile(r"^\d+_")
HEADER_ROW_PATTERN = re.compile(r"^[a-z]([a-z_0-9]+\s*)+$")
DATATYPE_ROW_PATTERN = re.compile(r"(\d+[sdn]\t?)+")
DATATYPE_PATTERN = re.compile(r"\d+(?P<datatype>[sdn])")
HEADING_EXPLANATION_PATTERN = re.compile(
    r"#\s+(?P<column>[a-z0-9_]+)\s+(\.\.\.|--)\s+(?P<description>[-A-Za-z _.\[()\]]+)"
)


DEFAULT_SITE_NAME = "Default"


T = typing.TypeVar("T")


DATATYPE_FUNCTIONS = {
    "s": str,
    "n": lambda val: float(val) if "." in val else int(val),
    "d": parse_date
}

DATATYPE_TYPES = {
    "s": str,
    "n": float,
}


class FrameTransformation:
    def __init__(
        self,
        column_name: str,
        transformation: typing.Callable[[...], typing.Any]
    ):
        if not isinstance(transformation, typing.Callable):
            raise TypeError(f"Transformation functions must be callables - received {type(transformation)} instead")

        self.__function = transformation
        self.__column_name = column_name

    def __call__(self, frame: pandas.DataFrame) -> pandas.DataFrame:
        frame[self.__column_name] = frame.apply(self.__function, axis=1)
        return frame


class NormalizeTimeZoneUnawareTransformation(FrameTransformation):
    def __init__(self, to_column_name: str, date_column: str, timezone_code_column):
        super().__init__(column_name=to_column_name, transformation=self.to_utc_unaware)
        self.__date_column = date_column
        self.__timezone_code_column = timezone_code_column

    def to_utc_unaware(self, row) -> pandas.Timestamp:
        date: pandas.Timestamp = row[self.__date_column]
        timezone_code = row[self.__timezone_code_column]
        timezone = gettz(timezone_code)
        localized_date = date.tz_localize(timezone)
        utc_date = localized_date.tz_convert(pytz.UTC)
        return utc_date.tz_localize(None)


class ToDayTransformation(FrameTransformation):
    def __init__(
        self,
        to_column_name: str,
        month_column: str = None,
        day_column: str = None,
        *,
        date_column: str = None
    ):
        if None in (month_column, day_column) and date_column is None:
            raise ValueError(
                f"Either a date column or month and day column are required in order to "
                f"transform values in a dataframe into a Day"
            )
        elif None not in (month_column, day_column) and date_column is not None:
            raise ValueError(
                f"A month and day column were passed into the Day transformation along with the name of the "
                f"date column. These are mutually exclusive"
            )
        super().__init__(column_name=to_column_name, transformation=self.to_day)
        self.__month_column = month_column
        self.__day_column = day_column
        self.__date_column = date_column

    def to_day(self, row: typing.Mapping) -> Day:
        if self.__month_column and self.__day_column:
            month_number = row[self.__month_column]
            day_number = row[self.__day_column]
            return Day(month_number=month_number, day_of_month_number=day_number)
        return Day(row[self.__date_column])


@dataclass
class RDBTable:
    parse_dates: typing.List[str] = field(default_factory=list)
    dtypes: typing.Dict[str, typing.Type] = field(default_factory=dict)
    data_lines: typing.List[str] = field(default_factory=list)
    columns: typing.Dict[str, str] = field(default_factory=dict)
    site_code: typing.Optional[str] = field(default=None)
    location_name: typing.Optional[str] = field(default=None)
    post_processing_functions: typing.List[FrameTransformation] = field(default_factory=list)

    def to_frame(self) -> pandas.DataFrame:
        buffer: ReadCsvBuffer[str] = io.StringIO(os.linesep.join(self.data_lines))

        creation_kwargs = {
            "na_values": [
                "Dis",          # Record has been discontinued at the measurement site.
                "Rat",          # Rating being developed
                "Mnt",          # Site undergoing maintenance
            ]
        }

        if self.dtypes:
            creation_kwargs['dtype'] = self.dtypes

        if self.parse_dates:
            creation_kwargs['parse_dates'] = self.parse_dates

        frame = pandas.read_csv(
            buffer,
            delimiter="\t",
            **creation_kwargs
        )

        for transformation in self.post_processing_functions:
            frame = transformation(frame)

        return frame


def extract_key_value_pairs(
    text: typing.Union[str, bytes, typing.Iterable[str]],
    pattern: re.Pattern
) -> typing.Dict[str, str]:
    if pattern.groups != 2:
        raise ValueError(
            f"There must be two groups in a pattern in order to extract key-value pairs. Received '{pattern.pattern}'"
        )

    if isinstance(text, bytes):
        text = text.decode()
    elif isinstance(text, typing.Iterable):
        text = os.linesep.join([str(line) for line in text])

    return {
        key: value
        for key, value in pattern.findall(text)
    }


class RDB:
    @classmethod
    def from_path(cls, path: pathlib.Path, frame_processing_functions: typing.Iterable[FrameTransformation] = None) -> RDB:
        text_data = path.read_text()
        return cls(text_data, frame_processing_functions)

    @classmethod
    def from_url(cls, url: str, frame_processing_functions: typing.Iterable[FrameTransformation] = None) -> RDB:
        with requests.get(url) as response:
            if response.status_code < 400:
                return cls(response.text, frame_processing_functions=frame_processing_functions)
            raise Exception(str(response.text))

    def __init__(self, text: typing.Union[str, bytes], frame_processing_functions: typing.Iterable[FrameTransformation] = None):
        self.__timeseries: typing.Dict[RDBTable] = dict()
        self.__default_data: typing.Optional[RDBTable] = None

        if frame_processing_functions is None:
            frame_processing_functions = []

        if isinstance(text, bytes):
            text = text.decode()

        locations: typing.Optional[typing.Dict[str, str]] = None

        columns: typing.List[str] = list()
        headings: typing.Dict[str, str] = dict()

        active_site = DEFAULT_SITE_NAME
        active_timeseries = None

        found_data = False
        for line in text.splitlines():
            if not found_data and SITE_CODE_PATTERN.match(line):
                match = SITE_CODE_PATTERN.match(line)
                if locations is None:
                    locations = {}

                locations[match.group("site_code")] = match.group("site_name")
                continue

            if SITE_TIMESERIES_ROW.match(line):
                match = SITE_TIMESERIES_ROW.match(line)
                active_site = match.group("site_code")
                continue

            if PARAMETER_CODE_PATTERN.match(line):
                match = PARAMETER_CODE_PATTERN.match(line)
                headings[match.group("pcode")] = match.group("description")
                continue

            if HEADER_ROW_PATTERN.match(line):
                columns = [
                    TS_ID_PREFIX_PATTERN.sub("", entry)
                    for entry in line.split("\t")
                ]
                found_data = False
                active_timeseries = RDBTable(
                    site_code=active_site,
                    location_name=locations.get(active_site) if locations and active_site in locations else None,
                    columns=headings,
                    post_processing_functions=frame_processing_functions
                )

                if active_site == DEFAULT_SITE_NAME:
                    self.__default_data = active_timeseries
                else:
                    self.__timeseries[active_site] = active_timeseries

                active_timeseries.data_lines.insert(0, "\t".join(columns))
                headings = {}
                continue

            if not found_data and HEADING_EXPLANATION_PATTERN.match(line):
                match = HEADING_EXPLANATION_PATTERN.match(line)
                headings[match.group("column")] = match.group("description")
                continue

            if DATATYPE_ROW_PATTERN.match(line):
                if found_data:
                    continue
                else:
                    datatypes = [
                        match.group("datatype")
                        for match in DATATYPE_PATTERN.finditer(line)
                    ]

                    for column, datatype in zip(columns, datatypes):
                        if datatype == 'd':
                            active_timeseries.parse_dates.append(column)
                        elif datatype in DATATYPE_TYPES:
                            active_timeseries.dtypes[column] = DATATYPE_TYPES[datatype]
                    continue

            if line.startswith("#"):
                continue

            found_data = True
            active_timeseries.data_lines.append(line)

    def __getitem__(self, key) -> RDBTable:
        return self.__timeseries[key]

    def __len__(self):
        return max(len(self.__timeseries), 1 if self.__default_data else 0)

    @property
    def default(self) -> typing.Optional[RDBTable]:
        return self.__default_data

    def get(self, key, __default: T = None) -> typing.Union[RDBTable, T]:
        if key == DEFAULT_SITE_NAME:
            return self.__default_data
        return self.__timeseries.get(key, __default)

    def items(self) -> ItemsView:
        return self.__timeseries.items()

    def keys(self) -> KeysView:
        return self.__timeseries.keys()

    def values(self) -> ValuesView[RDBTable]:
        return self.__timeseries.values()

    def __iter__(self):
        return iter(self.__timeseries)

