import logging
import os
import traceback
import unittest

from datetime import timedelta

import requests
from urllib3.exceptions import NameResolutionError
from urllib3.exceptions import MaxRetryError
from socket import gaierror
from requests.exceptions import ConnectionError

from randomod.data.rdb import RDB
from randomod.data.rdb import ToDayTransformation
from randomod.data.rdb import NormalizeTimeZoneUnawareTransformation

from test.utility import RESOURCE_DIRECTORY

TEST_TIMESERIES_PATH = RESOURCE_DIRECTORY / "alabama.rdb"
TEST_STATISTICS_PATH = RESOURCE_DIRECTORY / "statistics.rdb"

TEST_RDB_URL = "https://waterservices.usgs.gov/nwis/stat/?format=rdb&sites=02339495,02342500&statReportType=daily&statTypeCd=all"
RDB_CHECK_URL = "https://waterservices.usgs.gov"


STATISTICS_COLUMNS = [
    'agency_cd',
    'site_no',
    'parameter_cd',
    'ts_id',
    'loc_web_ds',
    'month_nu',
    'day_nu',
    'begin_yr',
    'end_yr',
    'count_nu',
    'max_va_yr',
    'max_va',
    'min_va_yr',
    'min_va',
    'mean_va',
    'p05_va',
    'p10_va',
    'p20_va',
    'p25_va',
    'p50_va',
    'p75_va',
    'p80_va',
    'p90_va',
    'p95_va'
]


class RDBTesting(unittest.TestCase):
    def test_timeseries_parsing(self):
        timeseries = RDB.from_path(TEST_TIMESERIES_PATH)

        self.assertEqual(len(timeseries), 229)  # add assertion here

        failing_location_messages = []

        for site_code, time_series in timeseries.items():
            try:
                frame = time_series.to_frame()
                del frame
            except BaseException as exception:
                failing_location_messages.append(f"Could not create a dataframe for {site_code}: {exception}")

        if failing_location_messages:
            message = f"Could not load all of the dataframes: {os.linesep}" + os.linesep.join(failing_location_messages)
            self.fail(message)

        utc_timeseries = RDB.from_path(
            TEST_TIMESERIES_PATH,
            frame_processing_functions=[
                NormalizeTimeZoneUnawareTransformation(
                    to_column_name="datetime",
                    date_column="datetime",
                    timezone_code_column="tz_cd"
                )
            ]
        )

        for site_code, time_series in utc_timeseries.items():
            non_utc_frame = timeseries[site_code].to_frame()

            try:
                frame = time_series.to_frame()
            except BaseException as exception:
                failing_location_messages.append(
                    f"Could not create and transform a dataframe for {site_code}: {exception}"
                )
                continue

            original_frame_was_cst = all((frame.datetime - non_utc_frame.datetime) == timedelta(hours=6))
            original_frame_was_cdt = all((frame.datetime - non_utc_frame.datetime) == timedelta(hours=5))

            self.assertTrue(original_frame_was_cst or original_frame_was_cdt)
            del frame

        if failing_location_messages:
            message = f"Could not load and transform all of the dataframes: {os.linesep}"
            message += os.linesep.join(failing_location_messages)
            self.fail(message)

    def test_statistics_parsing(self):
        rdb_text = TEST_STATISTICS_PATH.read_text()
        statistics = RDB(rdb_text)

        self.assertEqual(len(statistics), 1)

        failing_messages = []

        for site_code, statistics_data in statistics.items():
            try:
                frame = statistics_data.to_frame()
            except BaseException as exception:
                failing_messages.append(f"Could not interpret statistics: {exception}")
                continue

            self.assertEqual(len(frame.columns), len(STATISTICS_COLUMNS))

            for expected_column in STATISTICS_COLUMNS:
                self.assertIn(expected_column, frame.columns)

            self.assertEqual(frame.shape, (1464, 24))

            self.assertFalse(frame.loc_web_ds.any())
            self.assertTrue((frame.agency_cd == "USGS").all())
            del frame

        if failing_messages:
            message = f"Could not load all of the statistics RDB: {os.linesep}" + os.linesep.join(failing_messages)
            self.fail(message)

        statistics_with_day = RDB.from_path(
            TEST_STATISTICS_PATH,
            [
                ToDayTransformation(to_column_name="day", month_column="month_nu", day_column="day_nu")
            ]
        )

        self.assertEqual(len(statistics_with_day), 1)

        failing_messages = []

        for site_code, statistics_data in statistics_with_day.items():
            try:
                frame = statistics_data.to_frame()
            except BaseException as exception:
                failing_messages.append(f"Could not interpret statistics:{os.linesep}{traceback.format_exc()}")
                continue

            self.assertEqual(len(frame.columns), len(STATISTICS_COLUMNS) + 1)

            self.assertIn("day", frame.columns)

            for expected_column in STATISTICS_COLUMNS:
                self.assertIn(expected_column, frame.columns)

            self.assertEqual(frame.shape, (1464, 25))

            self.assertFalse(frame.loc_web_ds.any())
            self.assertTrue((frame.agency_cd == "USGS").all())
            del frame

        if failing_messages:
            message = f"Could not load and transform the statistics RDB: {os.linesep}" + os.linesep.join(failing_messages)
            self.fail(message)

    def test_url_parsing(self):
        try:
            requests.head(RDB_CHECK_URL)
        except (ConnectionError, gaierror, NameResolutionError, MaxRetryError) as connection_error:
            logging.error(connection_error)
            logging.warning(f"Cannot reach NWIS - bypassing the url parsing test for RDB")
            return

        rdb_data = RDB.from_url(
            TEST_RDB_URL,
            [
                ToDayTransformation(to_column_name="day", month_column="month_nu", day_column="day_nu")
            ]
        )

        self.assertEqual(len(rdb_data), 1)

        try:
            frame = rdb_data.default.to_frame()
        except BaseException as exception:
            raise Exception(f"Could not interpret statistics:{os.linesep}{traceback.format_exc()}") from exception

        self.assertEqual(len(frame.columns), len(STATISTICS_COLUMNS) + 1)

        self.assertIn("day", frame.columns)

        for expected_column in STATISTICS_COLUMNS:
            self.assertIn(expected_column, frame.columns)

        self.assertEqual(frame.shape, (1464, 25))

        self.assertFalse(frame.loc_web_ds.any())
        self.assertTrue((frame.agency_cd == "USGS").all())
        del frame


if __name__ == '__main__':
    unittest.main()
