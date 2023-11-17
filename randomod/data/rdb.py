"""
Defines classes and tools used to read and interpret USGS RDB files
"""
from __future__ import annotations

import io
import os
import typing
import re

import pandas

from dateutil.parser import parse as parse_date
from pandas._typing import ReadCsvBuffer
from pydantic import BaseModel
from pydantic import Field

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


DATATYPE_FUNCTIONS = {
    "s": str,
    "n": lambda val: float(val) if "." in val else int(val),
    "d": parse_date
}

DATATYPE_TYPES = {
    "s": str,
    "n": float,
}


class RDBTimeSeries(BaseModel):
    parse_dates: typing.List[str] = Field(default_factory=list)
    dtypes: typing.Dict[str, typing.Type] = Field(default_factory=dict)
    data_lines: typing.List[str] = Field(default_factory=list)
    columns: typing.Dict[str, str] = Field(default_factory=dict)
    site_code: typing.Optional[str] = Field(default=None)
    location_name: typing.Optional[str] = Field(default=None)

    def to_frame(self) -> pandas.DataFrame:
        buffer: ReadCsvBuffer[str] = io.StringIO(os.linesep.join(self.data_lines))

        creation_kwargs = {}

        if self.dtypes:
            creation_kwargs['dtype'] = self.dtypes

        if self.parse_dates:
            creation_kwargs['parse_dates'] = self.parse_dates

        return pandas.read_csv(
            buffer,
            delimiter="\t",
            **creation_kwargs
        )


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
    def parse(cls, text: typing.Union[str, bytes]):
        if isinstance(text, bytes):
            text = text.decode()

        timeseries: typing.Dict[str, RDBTimeSeries] = {}
        locations: typing.Optional[typing.Dict[str, str]] = None

        columns: typing.List[str] = list()
        headings: typing.Dict[str, str] = dict()

        active_site = "Unknown"
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
                active_timeseries = RDBTimeSeries(
                    site_code=active_site,
                    location_name=locations.get(active_site) if locations and active_site in locations else None,
                    columns=headings
                )
                timeseries[active_site] = active_timeseries
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
        return timeseries

    def __init__(self, text: str):
        pass